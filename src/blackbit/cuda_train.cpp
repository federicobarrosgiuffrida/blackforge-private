#include "blackforge/blackbit/cuda_train.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/checkpoint.hpp"
#include "blackforge/blackbit/cuda_checkpoint.hpp"
#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"
#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/blackbit/cuda_model.hpp"
#include "blackforge/data/dataset.hpp"
#include "blackforge/tokenizer/tokenizer.hpp"

namespace blackforge::blackbit::cuda {

namespace {

std::string gib(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)
        << " GiB";
    return out.str();
}

std::vector<int> integerValues(const runtime::Tensor& tensor, std::size_t vocabulary,
                               const char* role) {
    std::vector<int> result(tensor.elementCount());
    for (std::size_t index = 0; index < result.size(); ++index) {
        const float value = tensor.at(index);
        const int integer = static_cast<int>(std::lround(value));
        if (std::fabs(value - static_cast<float>(integer)) > 1.0e-5F || integer < 0 ||
            static_cast<std::size_t>(integer) >= vocabulary) {
            throw std::runtime_error(std::string("CUDA BlackBit trainer: invalid ") + role +
                                     " token in dataset at flat index " + std::to_string(index));
        }
        result[index] = integer;
    }
    return result;
}

std::uint64_t fileFnv1a64(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("CUDA BlackBit trainer: cannot read tokenizer '" + path + "'");
    std::uint64_t hash = 14695981039346656037ULL;
    char buffer[64 * 1024];
    while (input) {
        input.read(buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(buffer[index]);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::string jsonEscape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        result.push_back(ch);
    }
    return result;
}

std::string saveTrainingManifest(const std::string& checkpointPath, const TrainingOptions& options,
                                 const BlackBitConfig& config, std::size_t sequenceLength,
                                 const BlackBitTrainingState& state) {
    const std::string manifestPath = checkpointPath + ".metadata.json";
    std::string tokenizerCopy;
    std::size_t tokenizerVocab = 0;
    std::uint64_t tokenizerHash = 0;
    if (!options.tokenizerPath.empty()) {
        const tokenizer::Tokenizer tokenizer = tokenizer::loadTokenizer(options.tokenizerPath);
        tokenizerVocab = tokenizer.vocabSize();
        tokenizerHash = fileFnv1a64(options.tokenizerPath);
        tokenizerCopy = checkpointPath + ".bftok";
        std::filesystem::copy_file(options.tokenizerPath, tokenizerCopy,
                                   std::filesystem::copy_options::overwrite_existing);
    }
    std::ofstream out(manifestPath, std::ios::binary);
    if (!out) throw std::runtime_error("CUDA BlackBit trainer: cannot write checkpoint metadata");
    out << "{\n"
        << "  \"format\": \"blackforge-blackbit-training-metadata-v1\",\n"
        << "  \"checkpoint_format\": \"BFBIT-v" << kBlackBitCheckpointVersion << "\",\n"
        << "  \"step\": " << state.step << ",\n"
        << "  \"tokens_seen\": " << state.tokensSeen << ",\n"
        << "  \"dataset\": \"" << jsonEscape(options.datasetPath) << "\",\n"
        << "  \"dataset_fnv1a64\": \"" << std::hex << fileFnv1a64(options.datasetPath) << "\",\n"
        << "  \"sequence_length\": " << std::dec << sequenceLength << ",\n"
        << "  \"model_vocab_size\": " << config.vocabSize << ",\n"
        << "  \"tokenizer_format\": \"" << (tokenizerCopy.empty() ? "unspecified" : "BFTOKN1") << "\",\n"
        << "  \"tokenizer_copy\": \""
        << jsonEscape(tokenizerCopy.empty() ? std::string() : std::filesystem::path(tokenizerCopy).filename().string())
        << "\",\n"
        << "  \"tokenizer_vocab_size\": " << tokenizerVocab << ",\n"
        << "  \"tokenizer_fnv1a64\": \"" << std::hex << tokenizerHash << "\"\n"
        << "}\n";
    if (!out) throw std::runtime_error("CUDA BlackBit trainer: failed writing checkpoint metadata");
    return manifestPath;
}

}  // namespace

std::string TrainingResult::report() const {
    std::ostringstream out;
    out << "BlackBit CUDA real-text training\n";
    out << "  step                           " << initialStep << " -> " << finalStep << "\n";
    out << "  tokens                         " << initialTokens << " -> " << finalTokens << "\n";
    out << "  loss                           " << initialLoss << " -> " << finalLoss << "\n";
    out << "  perplexity                     " << std::exp(initialLoss) << " -> " << std::exp(finalLoss) << "\n";
    out << "  step time                      " << std::fixed << std::setprecision(2) << stepMilliseconds << " ms\n";
    out << "  throughput                     " << std::setprecision(3) << tokensPerSecond << " tokens/s\n";
    out << "  actual NVIDIA peak             " << gib(actualPeakBytes) << "\n";
    out << "  average NVIDIA used            " << gib(averageActualBytes) << "\n";
    out << "  ternary flips                  " << ternaryFlips << "\n";
    out << "  router entropy                 " << routingEntropy << " nat\n";
    out << "  max expert utilization         " << (100.0 * maxExpertUtilization) << " %\n";
    out << "  dropped assignments            " << droppedAssignments << "\n";
    out << "  NaN/Inf                        " << nanInfCount << "\n";
    if (!checkpointPath.empty()) {
        out << "  checkpoint                     " << checkpointPath << " (" << gib(checkpointBytes) << ")\n";
        if (!manifestPath.empty()) out << "  training metadata              " << manifestPath << "\n";
    }
    if (tokensPerSecond > 0.0) {
        constexpr double secondsPerYear = 365.25 * 24.0 * 3600.0;
        out << "  measured ETA 1B tokens         " << (1.0e9 / tokensPerSecond / secondsPerYear) << " years\n";
        out << "  measured ETA 10B tokens        " << (1.0e10 / tokensPerSecond / secondsPerYear) << " years\n";
        out << "  measured ETA 100B tokens       " << (1.0e11 / tokensPerSecond / secondsPerYear) << " years\n";
    }
    return out.str();
}

// Warmup lineare seguito da decadimento a coseno fino a minRatio del picco.
// Dipende dal passo GLOBALE, non da quello locale: cosi' una ripresa da
// checkpoint prosegue la stessa curva invece di ricominciarla.
float scheduledLearningRate(float peak, std::uint64_t globalStep, std::size_t warmupSteps,
                            std::size_t totalSteps, float minRatio) {
    if (warmupSteps > 0 && globalStep < warmupSteps) {
        return peak * static_cast<float>(globalStep + 1) / static_cast<float>(warmupSteps);
    }
    if (totalSteps <= warmupSteps) return peak;
    const double span = static_cast<double>(totalSteps - warmupSteps);
    const double done = static_cast<double>(globalStep - warmupSteps);
    const double progress = done >= span ? 1.0 : done / span;
    const double factor = 0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
    return peak * static_cast<float>(minRatio + (1.0 - minRatio) * factor);
}

TrainingResult train(const BlackBitConfig& config, const TrainingOptions& options, int device,
                     std::ostream* progress) {
    config.validate();
    if (options.datasetPath.empty() || options.steps == 0 || options.microBatch == 0) {
        throw std::invalid_argument("CUDA BlackBit trainer: dataset, positive steps and microbatch are required");
    }
    data::Dataset dataset = data::loadDataset(options.datasetPath);
    if (dataset.inputExampleShape().size() != 1 || dataset.targetExampleShape() != dataset.inputExampleShape()) {
        throw std::runtime_error("CUDA BlackBit trainer: dataset must contain matching rank-1 token/target sequences");
    }
    const std::size_t seq = dataset.inputExampleShape().front();
    if (seq > config.maxSeqLen) throw std::runtime_error("CUDA BlackBit trainer: dataset sequence exceeds model limit");
    if (!options.tokenizerPath.empty() &&
        tokenizer::loadTokenizer(options.tokenizerPath).vocabSize() > config.vocabSize) {
        throw std::runtime_error("CUDA BlackBit trainer: tokenizer vocabulary exceeds model vocabulary");
    }
    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    telemetry.initialize(device, options.maxVramMb == 0 ? 0 : options.maxVramMb * 1024ULL * 1024ULL);
    if (telemetry.currentTotal() == 0) telemetry.resetAccounting();

    blackforge::blackbit::BlackBitModel cpuReference(config, options.seed);
    cpuReference.setComputeDType(ComputeDType::BF16);
    cpuReference.setRuntimeOptions(options.runtime);
    BlackBitModel model(cpuReference);
    LowRankProjectedOptimizer optimizer(options.optimizer);
    model.registerParameters(optimizer);
    BlackBitTrainingState state;
    state.learningRate = options.optimizer.learningRate;
    state.rngSeed = options.seed;
    if (!options.fromCheckpoint.empty()) {
        state = cuda::loadCheckpoint(options.fromCheckpoint, model, &optimizer);
        optimizer.setLearningRate(state.learningRate);
    }
    TrainingResult result;
    result.initialStep = state.step;
    result.initialTokens = state.tokensSeen;
    result.lossCurve.reserve(options.steps);
    const std::size_t examples = dataset.numExamples();
    std::uint64_t currentEpoch = (state.step * options.microBatch) / examples;
    dataset.shuffle(static_cast<unsigned int>(state.rngSeed + currentEpoch));
    resetGradientLifetimeStats();
    telemetry.resetPeaks();
    double totalMilliseconds = 0.0;
    std::size_t memorySum = 0;
    const std::size_t scheduleSteps =
        options.scheduleSteps == 0 ? options.steps : options.scheduleSteps;
    for (std::size_t localStep = 0; localStep < options.steps; ++localStep) {
        state.learningRate =
            scheduledLearningRate(options.optimizer.learningRate, state.step, options.warmupSteps,
                                  scheduleSteps, options.minLearningRateRatio);
        optimizer.setLearningRate(state.learningRate);
        const std::uint64_t globalExample = state.step * options.microBatch;
        const std::size_t exampleIndex = static_cast<std::size_t>(globalExample % examples);
        const std::uint64_t epoch = globalExample / examples;
        if (epoch != currentEpoch) {
            currentEpoch = epoch;
            // Reload restores canonical order before applying this epoch's
            // deterministic permutation, making mid-run and resumed order identical.
            // Libera lo shard corrente PRIMA di caricare il successivo.
            //
            // L'assegnazione diretta costruisce la seconda copia mentre la prima
            // e' ancora viva: loadDataset alloca inputs e targets interi, li
            // riempie, e solo alla fine il risultato viene spostato dentro
            // 'dataset', distruggendo la vecchia. Il picco e' quindi due copie.
            //
            // Su questo shard una copia e' 1.056.143 x 1024 x 4 byte x 2 vettori
            // = 8,65 GB, quindi il picco e' 17,3 GB contro 18,8 GB liberi sulla
            // macchina: un margine dell'8 %, e un fallimento qui sarebbe fra i
            // peggiori possibili, perche' e' DETERMINISTICO. Ripartire dal
            // checkpoint riporta al confine di epoca, che rifallisce, in un
            // ciclo che lascia la macchina ferma finche' qualcuno non guarda.
            //
            // Svuotare prima dimezza il picco a una copia sola. Costa una
            // rilettura da disco che stiamo gia' facendo comunque.
            dataset = data::Dataset(std::vector<std::size_t>{}, std::vector<std::size_t>{},
                                    std::vector<float>{}, std::vector<float>{}, 0);
            dataset = data::loadDataset(options.datasetPath);
            dataset.shuffle(static_cast<unsigned int>(state.rngSeed + currentEpoch));
        }
        const data::Dataset::Batch batch = dataset.batch(exampleIndex, options.microBatch);
        const std::vector<int> inputs = integerValues(batch.input, config.vocabSize, "input");
        const std::vector<int> targets = integerValues(batch.target, config.vocabSize, "target");
        BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
        const auto before = std::chrono::steady_clock::now();
        const BlackBitStepResult step = model.trainStep(inputs, targets, options.microBatch, seq, &optimizer);
        optimizer.endStep();
        BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
        const auto after = std::chrono::steady_clock::now();
        const double milliseconds = std::chrono::duration<double, std::milli>(after - before).count();
        totalMilliseconds += milliseconds;
        result.lossCurve.push_back(step.loss);
        if (localStep == 0) result.initialLoss = step.loss;
        result.finalLoss = step.loss;
        result.routingEntropy = step.meanRoutingEntropy();
        result.maxExpertUtilization = step.maxExpertUtilization();
        result.droppedAssignments = step.droppedAssignments();
        result.nanInfCount += optimizer.stats().lastStep.nanInfCount + (step.sawNaN || step.sawInf ? 1U : 0U);
        ++state.step;
        state.tokensSeen += options.microBatch * seq;
        state.optimizerStep = optimizer.stepCount();
        const DeviceMemorySnapshot memory = telemetry.snapshot();
        memorySum += memory.usedBytes;
        if (progress != nullptr) {
            // Le statistiche dell'ottimizzatore erano raccolte a ogni passo e
            // stampate soltanto nel rapporto finale. Durante un run lungo non
            // se ne vedeva una, e davanti a una loss ferma non c'era modo di
            // distinguere fra le cause possibili: aggiornamenti troppo
            // piccoli per far scattare i trit, trit saturati che non possono
            // piu' muoversi, gradiente collassato, o rapporto
            // aggiornamento/peso troppo basso. Sono quattro ipotesi diverse
            // con quattro rimedi diversi, e i numeri per separarle erano gia'
            // in memoria.
            const auto& st = optimizer.stats().lastStep;
            *progress << "  text step " << state.step << ": loss " << step.loss << ", "
                      << (1000.0 * options.microBatch * seq / milliseconds) << " token/s, peak "
                      << gib(memory.devicePeakUsedBytes)
                      << ", flip " << st.flips
                      << " (" << std::setprecision(4) << (st.flipFraction() * 100.0) << "%)"
                      << ", sat " << st.saturatedTrits
                      << ", gradRms " << st.gradientRms
                      << ", updRms " << st.updateRms
                      << ", optNorm " << st.optimizerNorm << "\n";
        }
        if (result.nanInfCount != 0) throw std::runtime_error("CUDA BlackBit trainer: numerical instability");
        if (options.checkpointEvery != 0 && !options.saveCheckpoint.empty() &&
            (localStep + 1) % options.checkpointEvery == 0 && localStep + 1 != options.steps) {
            // Scrittura su temporaneo e rinomina: se il processo muore a meta'
            // della scrittura, il checkpoint precedente resta intatto e la
            // rinomina non e' mai osservata a meta'.
            const std::filesystem::path target(options.saveCheckpoint);
            std::filesystem::path temporary = target;
            temporary += ".tmp";
            cuda::saveCheckpoint(temporary.string(), model, state, &optimizer);
            std::filesystem::rename(temporary, target);
            saveTrainingManifest(options.saveCheckpoint, options, config, seq, state);
            if (progress != nullptr) {
                *progress << "  checkpoint al passo " << state.step << ": " << options.saveCheckpoint
                          << "\n";
            }
        }
    }
    result.finalStep = state.step;
    result.finalTokens = state.tokensSeen;
    result.stepMilliseconds = totalMilliseconds / options.steps;
    result.tokensPerSecond = static_cast<double>(options.steps * options.microBatch * seq) /
                             (totalMilliseconds / 1000.0);
    result.ternaryFlips = optimizer.stats().totalFlips;
    result.actualPeakBytes = telemetry.snapshot().devicePeakUsedBytes;
    result.averageActualBytes = memorySum / options.steps;
    if (!options.saveCheckpoint.empty()) {
        cuda::saveCheckpoint(options.saveCheckpoint, model, state, &optimizer);
        result.checkpointPath = options.saveCheckpoint;
        result.checkpointBytes = static_cast<std::size_t>(std::filesystem::file_size(options.saveCheckpoint));
        result.manifestPath = saveTrainingManifest(options.saveCheckpoint, options, config, seq, state);
    }
    return result;
}

EvalResult evaluate(const EvalOptions& options, int device, std::ostream* progress) {
    if (options.checkpoint.empty() || options.datasetPath.empty() || options.microBatch == 0) {
        throw std::invalid_argument("CUDA BlackBit eval: servono checkpoint, dataset e microbatch positivo");
    }
    const BlackBitConfig config = readCheckpointConfig(options.checkpoint);
    data::Dataset dataset = data::loadDataset(options.datasetPath);
    if (dataset.inputExampleShape().size() != 1 || dataset.targetExampleShape() != dataset.inputExampleShape()) {
        throw std::runtime_error("CUDA BlackBit eval: il dataset deve contenere sequenze di rango 1");
    }
    const std::size_t seq = dataset.inputExampleShape().front();
    if (seq > config.maxSeqLen) throw std::runtime_error("CUDA BlackBit eval: sequenza oltre il limite del modello");

    MemoryTelemetry& telemetry = MemoryTelemetry::instance();
    telemetry.initialize(device, options.maxVramMb == 0 ? 0 : options.maxVramMb * 1024ULL * 1024ULL);

    blackforge::blackbit::BlackBitModel cpuReference(config, options.seed);
    cpuReference.setComputeDType(ComputeDType::BF16);
    cpuReference.setRuntimeOptions(options.runtime);
    BlackBitModel model(cpuReference);
    const BlackBitTrainingState state = loadCheckpoint(options.checkpoint, model);

    // Il dataset NON viene rimescolato: la valutazione deve dare lo stesso
    // numero sugli stessi esempi a ogni chiamata, altrimenti confrontare due
    // checkpoint misura anche la differenza fra i due campioni.
    std::size_t total = dataset.numExamples();
    if (options.maxExamples > 0 && options.maxExamples < total) total = options.maxExamples;
    const std::size_t batches = total / options.microBatch;   // la coda parziale si scarta
    if (batches == 0) throw std::runtime_error("CUDA BlackBit eval: il dataset non copre un microbatch");

    EvalResult result;
    result.checkpointStep = state.step;
    result.checkpointTokens = state.tokensSeen;
    result.examples = batches * options.microBatch;

    double sommaNats = 0.0;   // pesata sui token effettivamente valutati
    std::uint64_t scored = 0;
    const auto inizio = std::chrono::steady_clock::now();
    for (std::size_t b = 0; b < batches; ++b) {
        const data::Dataset::Batch batch = dataset.batch(b * options.microBatch, options.microBatch);
        const std::vector<int> inputs = integerValues(batch.input, config.vocabSize, "input");
        const std::vector<int> targets = integerValues(batch.target, config.vocabSize, "target");
        // sink nullo: trainStep calcola la loss in avanti e ritorna prima del
        // backward, quindi nessun parametro viene toccato.
        const BlackBitStepResult step = model.trainStep(inputs, targets, options.microBatch, seq, nullptr);
        if (step.sawNaN || step.sawInf) {
            throw std::runtime_error("CUDA BlackBit eval: NaN/Inf su '" + step.firstUnstableTensor + "'");
        }
        sommaNats += static_cast<double>(step.loss) * static_cast<double>(step.scoredTokens);
        scored += step.scoredTokens;
        if (progress != nullptr && (b + 1) % 50 == 0) {
            *progress << "  valutati " << (b + 1) << "/" << batches << " microbatch\r" << std::flush;
        }
    }
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
    const auto fine = std::chrono::steady_clock::now();
    result.seconds = std::chrono::duration<double>(fine - inizio).count();

    result.scoredTokens = scored;
    result.loss = scored == 0 ? 0.0 : sommaNats / static_cast<double>(scored);
    result.perplexity = std::exp(result.loss);
    if (options.bytesPerToken > 0.0) {
        // bit/byte = (nats/token) / (ln2 * byte/token)
        result.bitsPerByte = result.loss / (0.6931471805599453 * options.bytesPerToken);
    }
    return result;
}

}  // namespace blackforge::blackbit::cuda
