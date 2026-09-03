#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <string>
#include <vector>

#include "blackforge/ast/ast.hpp"
#include "blackforge/blackbit/benchmark.hpp"
#include "blackforge/blackbit/checkpoint.hpp"
#include "blackforge/blackbit/config.hpp"
#include "blackforge/backend/cpu/benchmark.hpp"
#include "blackforge/backend/cpu/checkpoint.hpp"
#include "blackforge/backend/cpu/executor.hpp"
#include "blackforge/backend/cpu/forecast_runner.hpp"
#include "blackforge/backend/cpu/model.hpp"
#include "blackforge/backend/cpu/train_runner.hpp"
#include "blackforge/diagnostics/diagnostic.hpp"
#include "blackforge/frontend/lexer.hpp"
#include "blackforge/frontend/parser.hpp"
#include "blackforge/frontend/token.hpp"
#include "blackforge/ir/ir_builder.hpp"
#include "blackforge/ir/module.hpp"
#include "blackforge/runtime/tensor.hpp"
#include "blackforge/sema/dtype.hpp"
#include "blackforge/sema/semantic_analyzer.hpp"
#include "blackforge/tokenizer/dataset_prep.hpp"
#include "blackforge/tokenizer/tokenizer.hpp"

#if BLACKFORGE_HAS_CUDA
#include "blackforge/backend/cuda/checkpoint.hpp"
#include "blackforge/backend/cuda/device_query.hpp"
#include "blackforge/backend/cuda/executor.hpp"
#include "blackforge/backend/cuda/model.hpp"
#include "blackforge/backend/cuda/multi_gpu_train_runner.hpp"
#include "blackforge/backend/cuda/train_runner.hpp"
#include "blackforge/blackbit/cuda_benchmark.hpp"
#include "blackforge/blackbit/cuda_profile.hpp"
#include "blackforge/blackbit/cuda_train.hpp"
#endif

namespace {

void printUsage() {
    std::cout << "Uso: blackforge <comando> [opzioni] <file.bf>\n\n"
              << "Comandi:\n"
              << "  check <file>       Analizza il file e riporta gli errori (lessicali, sintattici, semantici)\n"
              << "  build <file>       Compila e costruisce ogni modello (alloca i parametri), senza eseguirlo\n"
              << "  run <file>         Esegue il primo modello del file\n"
              << "  train <file>       Addestra il modello descritto dal blocco 'train' del file (CPU o CUDA)\n"
              << "  train blackbit [preset]  Addestra BlackBit CUDA su uno shard .bfdata reale\n"
              << "  forecast <file>    Genera 'horizon' passi autoregressivi dal blocco 'forecast' del file (CPU)\n"
              << "  benchmark <file>   Misura tempo/throughput/memoria del primo modello del file\n"
              << "  benchmark blackbit [preset]   Misura un modello BlackBit (MoE ternario a bassa memoria):\n"
              << "                                  parametri totali/attivi, memoria per arena, tempi, e le\n"
              << "                                  verifiche FULL PRECISION MASTER COPY / FULL MODEL GRADIENT\n"
              << "                                  BUFFER. Preset: tiny, small, medium, 9b-a3b; oppure --config\n"
              << "  inspect <file>     Mostra un riepilogo dei modelli (input, pipeline, numero di parametri)\n"
              << "  devices            Elenca i dispositivi di calcolo disponibili (CPU e GPU CUDA)\n"
              << "  blackbit-generate  Genera testo da un checkpoint BlackBit (richiede --from-checkpoint e\n"
              << "                     --tokenizer; opzionali --prompt, --max-new-tokens, --temperature).\n"
              << "                     Temperatura 0 sceglie sempre il token piu' probabile\n"
              << "  --top-p F          Nucleus sampling: tiene il minimo insieme di token la cui massa cumulata\n"
              << "                     raggiunge F, scartando la coda (default 0.9; 1 disattiva)\n"
              << "  --top-k N          Tiene solo gli N token piu probabili (default 0 = disattivo)\n"
              << "  --repetition-penalty F  Allontana dai token gia generati nella finestra recente. Serve anche a\n"
              << "                     temperatura 0, dove la decodifica greedy nuda cade in cicli (default 1.1)\n"
              << "  --penalty-window N Quanti token recenti considera la penalita (default 64; 0 = tutti)\n"
              << "  blackbit-eval      Valuta un checkpoint su dati mai visti e riporta loss, perplexity e bit\n"
              << "                     per byte (richiede --from-checkpoint, --dataset e --device cuda;\n"
              << "                     --bytes-per-token abilita i bit per byte, --max-examples limita)\n"
              << "  tokenizer-train <corpus.txt>   Addestra un tokenizer BPE byte-level sul corpus dato\n"
              << "                                  (richiede --vocab-size e --output)\n"
              << "  tokenizer-encode <tok.bftok> <testo.txt>   Codifica un file di testo, stampa gli id di token\n"
              << "  dataset-build <corpus.txt> <tok.bftok>   Tokenizza un corpus e costruisce un dataset .bfdata\n"
              << "                                  per l'addestramento di un modello linguistico causale\n"
              << "                                  (richiede --seq-len e --output; aggiungi --mlm per un\n"
              << "                                  dataset mascherato stile BERT invece, con --mask-prob)\n"
              << "  generate <file>     Genera testo autoregressivamente con cache K/V (CPU o GPU via --device,\n"
              << "                                  greedy/argmax, richiede --from-checkpoint, --tokenizer, "
                 "--prompt)\n"
              << "  --help, -h         Mostra questo messaggio\n"
              << "  --version, -v      Mostra la versione del compilatore\n\n"
              << "Opzioni:\n"
              << "  --verbose          Mostra i token riconosciuti (solo 'check')\n"
              << "  --print-ast        Mostra l'albero sintattico (AST) prodotto dal parser (solo 'check')\n"
              << "  --print-ir         Mostra la rappresentazione interna (IR) del programma (solo 'check')\n"
              << "  --batch N          Dimensione di batch usata per risolvere le dimensioni simboliche "
                 "dell'input (default 1)\n"
              << "  --device cpu|cuda|cuda:N  Dispositivo su cui eseguire ('run'/'train'/'generate'/'benchmark', "
                 "default cpu). Su 'train', il backend CUDA supporta 'loss mse' e 'cross_entropy' ma non ancora "
                 "'lora' (errore esplicito se richiesto); checkpoint supportati su entrambi i device, stesso "
                 "formato\n"
              << "  --devices 0,1,...  Addestramento data-parallelo su piu' GPU CUDA (solo 'train', alternativo "
                 "a '--device'; almeno 2 indici, 'batch_size' deve essere divisibile per il numero di GPU); "
                 "all-reduce dei gradienti via staging host (NCCL non e' disponibile su Windows)\n"
              << "  --from-checkpoint <file>  Pesi da caricare: esecuzione con pesi allenati per 'run', "
                 "fine-tuning/LoRA per 'train' (LoRA solo CPU), obbligatorio per 'forecast' (solo CPU)\n"
              << "  --save-checkpoint <file>  Dove salvare i pesi al termine dell'addestramento (solo 'train')\n"
              << "  --warmup N         Iterazioni di riscaldamento scartate (solo 'benchmark', default 5)\n"
              << "  --iterations N     Iterazioni misurate (solo 'benchmark', default 20)\n"
              << "  --vocab-size N     Dimensione del vocabolario da apprendere (solo 'tokenizer-train'; deve "
                 "superare " << blackforge::tokenizer::Tokenizer::kFirstMergeId << ")\n"
              << "  --seq-len N        Lunghezza di sequenza di ogni esempio (solo 'dataset-build')\n"
              << "  --output <file>    Percorso di output (solo 'tokenizer-train'/'dataset-build')\n"
              << "  --tokenizer <file> Tokenizer .bftok per 'generate' o metadati checkpoint BlackBit\n"
              << "  --prompt <testo>   Testo iniziale da cui continuare la generazione (solo 'generate')\n"
              << "  --max-new-tokens N Numero massimo di token da generare (solo 'generate', default 50)\n"
              << "  --mlm              Costruisce un dataset mascherato (MLM) invece che causale (solo "
                 "'dataset-build')\n"
              << "  --mask-prob P      Probabilita' di mascherare ogni posizione, 0 < P <= 1 (solo 'dataset-build "
                 "--mlm', default 0.15)\n"
              << "  --config <file>    Configurazione JSON di BlackBit (solo 'benchmark blackbit')\n"
              << "  --dataset <file>   Shard tokenizzato .bfdata (solo 'train blackbit')\n"
              << "  --micro-batch N    Sequenze per passo (solo 'benchmark blackbit', default 1)\n"
              << "  --steps N          Passi di addestramento misurati (solo 'benchmark blackbit', default 3)\n"
              << "  --instantiate-only Costruisce BlackBit CUDA e misura la VRAM senza eseguire uno step\n"
              << "  --profile          Attribuisce il tempo GPU per fase con cudaEvent (solo 'benchmark "
                 "blackbit' su CUDA)\n"
              << "  --seq-ladder      Esegue seq 8,16,...,512 sulla stessa istanza BlackBit CUDA\n"
              << "  --max-vram-mb N    Budget di memoria: superarlo e' un errore diagnosticato prima di allocare "
                 "(solo 'benchmark blackbit', default 7800, 0 = nessun limite)\n"
              << "  --optimizer-rank N Rango del sottospazio dell'ottimizzatore (solo 'benchmark blackbit')\n"
              << "  --latent-decay F   Decadimento applicato al peso latente a ogni passo con --error-feedback.\n"
              << "                     Impedisce ai pesi di accumularsi al bordo del limite, dove smettono di\n"
              << "                     rispondere al gradiente (default 0)\n"
              << "  --latent-clamp F   Ampiezza del limite sul peso latente con --error-feedback (default 1).\n"
              << "                     Valori maggiori restituiscono inerzia contro il rumore del gradiente,\n"
              << "                     tenendo comunque il residuo limitato\n"
              << "  --error-feedback   Accumula il residuo di arrotondamento invece di scartarlo: il peso\n"
              << "                     latente diventa trit + residuo e l'arrotondamento passa da stocastico\n"
              << "                     a deterministico. Costa un float per peso (solo 'train blackbit')\n"
              << "  --checkpoint-every N  Salva un checkpoint ogni N passi, non solo alla fine. Scrittura su\n"
              << "                     temporaneo e rinomina: un'interruzione non corrompe il file buono\n"
              << "                     (solo 'train blackbit', default 0 = solo alla fine)\n"
              << "  --learning-rate F  Learning rate di picco (solo 'train blackbit')\n"
              << "  --warmup-steps N   Passi di warmup lineare prima del coseno (solo 'train blackbit')\n"
              << "  --schedule-steps N Passi totali PIANIFICATI su cui stendere il coseno; riprendendo da un\n"
              << "                     checkpoint va indicato il totale dell'intero addestramento, non i passi\n"
              << "                     di questa invocazione (solo 'train blackbit', default --steps)\n"
              << "  --recompute MODO   none | per-layer | every-N | full: quante attivazioni conservare invece "
                 "di ricalcolarle (solo 'benchmark blackbit', default per-layer)\n"
              << "  --dry-run          Riporta solo la contabilita' prevista, senza allocare nulla (solo "
                 "'benchmark blackbit')\n";
}

void printDevices() {
    std::cout << "cpu: sempre disponibile (backend di riferimento)\n";

#if BLACKFORGE_HAS_CUDA
    auto devices = blackforge::backend::cuda::enumerateDevices();
    if (devices.empty()) {
        std::cout << "cuda: nessuna GPU NVIDIA rilevata (compilato con supporto CUDA, ma nessun dispositivo "
                     "visibile al driver)\n";
    }
    for (const auto& device : devices) {
        std::cout << "cuda:" << device.index << ": " << device.name << " (compute capability "
                   << device.computeCapabilityMajor << "." << device.computeCapabilityMinor << ", "
                   << (device.totalMemoryBytes / (1024ULL * 1024ULL)) << " MiB)\n";
    }
#else
    std::cout << "cuda: non disponibile (questa build e' stata compilata senza supporto CUDA)\n";
#endif
}

void printVersion() { std::cout << "blackforge " << BLACKFORGE_VERSION << "\n"; }

struct DeviceSpec {
    bool isCuda = false;
    int cudaIndex = 0;
};

// Analizza '--devices 0,1,2': un elenco di indici GPU separati da
// virgola, per l'addestramento data-parallelo multi-GPU (vedi
// blackforge::backend::cuda::runMultiGpuTraining). Restituisce false e
// riempie 'error' se la stringa e' vuota, contiene un elemento non
// numerico, o ha meno di 2 indici (per 1 sola GPU si usa '--device
// cuda:N', non questo flag).
bool parseDeviceList(const std::string& devicesList, std::vector<int>& outIndices, std::string& outError) {
    outIndices.clear();
    std::string current;
    auto flush = [&](const std::string& token) -> bool {
        if (token.empty() ||
            !std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            outError = "indice GPU non valido in '--devices " + devicesList + "'";
            return false;
        }
        outIndices.push_back(std::stoi(token));
        return true;
    };
    for (char c : devicesList) {
        if (c == ',') {
            if (!flush(current)) {
                return false;
            }
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!flush(current)) {
        return false;
    }
    if (outIndices.size() < 2) {
        outError = "'--devices' richiede almeno 2 indici GPU separati da virgola (per 1 sola GPU usa '--device "
                   "cuda:N')";
        return false;
    }
    return true;
}

// Analizza '--device cpu' | 'cuda' | 'cuda:N'. Restituisce false e
// riempie 'error' se la stringa non e' un dispositivo valido.
bool parseDeviceSpec(const std::string& device, DeviceSpec& outSpec, std::string& outError) {
    if (device == "cpu") {
        outSpec = DeviceSpec{false, 0};
        return true;
    }
    if (device == "cuda") {
        outSpec = DeviceSpec{true, 0};
        return true;
    }
    if (device.rfind("cuda:", 0) == 0) {
        std::string indexPart = device.substr(5);
        bool allDigits = !indexPart.empty() && std::all_of(indexPart.begin(), indexPart.end(), [](unsigned char c) {
                              return std::isdigit(c) != 0;
                          });
        if (!allDigits) {
            outError = "indice GPU non valido in '" + device + "'";
            return false;
        }
        outSpec = DeviceSpec{true, std::stoi(indexPart)};
        return true;
    }
    outError = "dispositivo sconosciuto '" + device + "' (valori validi: cpu, cuda, cuda:N)";
    return false;
}

// Legge l'intero contenuto di un file in una stringa. Restituisce false
// se il file non esiste o non e' leggibile.
bool readFile(const std::string& path, std::string& outContent) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    outContent = buffer.str();
    return true;
}

enum class CompileStatus { FileNotFound, AnalysisFailed, Ok };

struct CompileOutput {
    CompileStatus status = CompileStatus::FileNotFound;
    blackforge::ast::Program program;
    blackforge::ir::Module module;
    std::size_t tokenCount = 0;
};

// Esegue lexer, parser, analisi semantica e costruzione della IR su un
// file, stampando tutte le diagnostiche su stderr. La IR viene
// costruita solo se le fasi precedenti non hanno trovato errori.
CompileOutput compile(const std::string& path, bool verbose, bool printAst, bool printIr) {
    CompileOutput result;

    std::string source;
    if (!readFile(path, source)) {
        std::cerr << "errore: impossibile aprire il file '" << path << "'\n";
        return result;
    }

    blackforge::Lexer lexer(source, path);
    std::vector<blackforge::Token> tokens = lexer.tokenize();
    result.tokenCount = tokens.size();

    if (verbose) {
        for (const auto& token : tokens) {
            std::cout << token.location.file << ":" << token.location.line << ":" << token.location.column << ": "
                      << blackforge::tokenKindName(token.kind);
            if (!token.lexeme.empty()) {
                std::cout << " '" << token.lexeme << "'";
            }
            std::cout << "\n";
        }
    }

    blackforge::Parser parser(tokens);
    result.program = parser.parseProgram();

    if (printAst) {
        std::cout << blackforge::ast::dump(result.program);
    }

    blackforge::sema::SemanticAnalyzer analyzer;
    analyzer.analyze(result.program);

    bool hasErrors = false;
    for (const auto& diagnostic : lexer.diagnostics().all()) {
        std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
    }
    hasErrors = hasErrors || lexer.diagnostics().hasErrors();

    for (const auto& diagnostic : parser.diagnostics().all()) {
        std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
    }
    hasErrors = hasErrors || parser.diagnostics().hasErrors();

    for (const auto& diagnostic : analyzer.diagnostics().all()) {
        std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
    }
    hasErrors = hasErrors || analyzer.diagnostics().hasErrors();

    if (!hasErrors) {
        blackforge::ir::IRBuilder irBuilder;
        result.module = irBuilder.build(result.program);

        if (printIr) {
            std::cout << blackforge::ir::dump(result.module);
        }

        for (const auto& diagnostic : irBuilder.diagnostics().all()) {
            std::cerr << blackforge::formatDiagnostic(diagnostic) << "\n";
        }
        hasErrors = hasErrors || irBuilder.diagnostics().hasErrors();
    }

    result.status = hasErrors ? CompileStatus::AnalysisFailed : CompileStatus::Ok;
    return result;
}

int runCheck(const std::string& path, bool verbose, bool printAst, bool printIr) {
    CompileOutput result = compile(path, verbose, printAst, printIr);

    switch (result.status) {
        case CompileStatus::FileNotFound:
            return 2;
        case CompileStatus::AnalysisFailed:
            std::cerr << path << ": analisi fallita\n";
            return 1;
        case CompileStatus::Ok:
            std::cout << path << ": nessun errore (" << result.tokenCount << " token, "
                       << result.program.declarations.size() << " dichiarazioni top-level)\n";
            return 0;
    }
    return 1;
}

int runRun(const std::string& path, std::size_t batchSize, const std::string& device,
           const std::string& fromCheckpoint) {
    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile eseguire, l'analisi ha trovato errori\n";
        return 1;
    }
    if (result.module.models.empty()) {
        std::cerr << "errore: il programma non definisce alcun modello da eseguire\n";
        return 1;
    }

    const blackforge::ir::ModelIR& model = result.module.models.front();

    if (!spec.isCuda && result.module.target.has_value() && *result.module.target != "cpu") {
        std::cout << "nota: target dichiarato '" << *result.module.target
                   << "'; sto eseguendo su CPU. Usa '--device cuda' per eseguire sulla GPU (se disponibile).\n";
    }

    try {
        blackforge::runtime::Tensor input;
        blackforge::runtime::Tensor output;

        if (!spec.isCuda) {
            blackforge::backend::cpu::Executor executor;
            input = executor.makeSyntheticInput(model, batchSize);

            if (fromCheckpoint.empty()) {
                output = executor.run(model, input, result.module.precision);
            } else {
                // A differenza di Executor (pesi casuali rigenerati ad ogni
                // chiamata), Model possiede i propri parametri: e' quello che
                // serve per eseguire davvero con pesi allenati caricati da
                // un checkpoint, non solo per verificare che la pipeline
                // funzioni.
                blackforge::backend::cpu::Model loadedModel(model, /*seed=*/42, /*lora=*/std::nullopt,
                                                              result.module.precision);
                blackforge::backend::cpu::loadCheckpoint(loadedModel, fromCheckpoint);
                std::cout << "Pesi caricati da '" << fromCheckpoint << "'\n";
                output = loadedModel.forward(input);
            }
        } else {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);
            blackforge::backend::cuda::Executor executor;
            input = executor.makeSyntheticInput(model, batchSize);

            if (fromCheckpoint.empty()) {
                output = executor.run(model, input);
            } else {
                blackforge::backend::cuda::Model loadedModel(model, /*seed=*/42, result.module.precision);
                blackforge::backend::cuda::loadCheckpoint(loadedModel, fromCheckpoint);
                std::cout << "Pesi caricati da '" << fromCheckpoint << "'\n";
                blackforge::backend::cuda::DeviceTensor inputDevice =
                    blackforge::backend::cuda::DeviceTensor::fromHost(input);
                output = loadedModel.forward(inputDevice).toHost();
            }
            if (result.module.precision.has_value() &&
                result.module.precision->compute == blackforge::sema::DType::BF16 && !fromCheckpoint.empty()) {
                std::cout << "nota: 'precision { compute bf16 }' su GPU usa Tensor Core BF16 reali per ogni "
                             "layer 'linear' (non quantizzazione simulata); 'storage'/'accumulate' restano "
                             "ignorati (sempre float32).\n";
            } else if (result.module.precision.has_value()) {
                std::cout << "nota: un blocco 'precision' e' dichiarato, ma il backend CUDA applica solo "
                             "'compute bf16' (Tensor Core reale, solo con --from-checkpoint) — "
                             "'storage'/'accumulate' e qualunque altro 'compute' restano ignorati (sempre "
                             "float32 su GPU).\n";
            }
#endif
        }

        std::cout << "Eseguito modello '" << model.name << "' sul backend " << device << "\n";
        std::cout << "  input:  " << input.shapeToString() << "\n";
        std::cout << "  output: " << output.shapeToString() << "  min=" << output.min() << " max=" << output.max()
                   << " mean=" << output.mean() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di esecuzione: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runTrain(const std::string& path, const std::string& fromCheckpoint, const std::string& saveCheckpointPath,
             const std::string& device, const std::string& devicesList) {
    if (!devicesList.empty()) {
        // '--devices' e' esclusivo di CUDA multi-GPU: non ha senso
        // insieme a '--device cpu' (nessun training multi-CPU) ne'
        // insieme a '--device cuda:N' (un solo indice, ambiguo con
        // l'elenco di '--devices').
        std::vector<int> deviceIndices;
        std::string devicesError;
        if (!parseDeviceList(devicesList, deviceIndices, devicesError)) {
            std::cerr << "errore: " << devicesError << "\n";
            return 2;
        }
        if (!BLACKFORGE_HAS_CUDA) {
            std::cerr << "errore: '--devices' richiesto ma questa build di blackforge e' stata compilata senza "
                         "supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
            return 1;
        }

        CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
        if (result.status == CompileStatus::FileNotFound) {
            return 2;
        }
        if (result.status == CompileStatus::AnalysisFailed) {
            std::cerr << path << ": impossibile addestrare, l'analisi ha trovato errori\n";
            return 1;
        }

        try {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::runMultiGpuTraining(result.program, result.module, deviceIndices,
                                                            fromCheckpoint, saveCheckpointPath, &std::cout);
#endif
        } catch (const std::exception& e) {
            std::cerr << "errore di addestramento multi-GPU: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile addestrare, l'analisi ha trovato errori\n";
        return 1;
    }

    try {
        if (!spec.isCuda) {
            blackforge::backend::cpu::runTraining(result.program, result.module, fromCheckpoint, saveCheckpointPath,
                                                   &std::cout);
        } else {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);
            blackforge::backend::cuda::runTraining(result.program, result.module, fromCheckpoint, saveCheckpointPath,
                                                    &std::cout);
#endif
        }
    } catch (const std::exception& e) {
        std::cerr << "errore di addestramento: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runForecast(const std::string& path, const std::string& fromCheckpoint, std::size_t batchSize) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);

    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile eseguire il forecasting, l'analisi ha trovato errori\n";
        return 1;
    }

    try {
        blackforge::backend::cpu::runForecast(result.program, result.module, fromCheckpoint, batchSize, &std::cout);
    } catch (const std::exception& e) {
        std::cerr << "errore di forecasting: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Sceglie il prossimo id in base ai logit dell'ultima posizione di un
// tensore host [1, seq, vocabSize]: decodifica sempre greedy (argmax),
// nessuna strategia di campionamento (temperatura/top-k/nucleus) ancora
// implementata — una limitazione esplicita, non nascosta: la stessa
// sequenza in ingresso produce sempre la stessa continuazione.
std::uint32_t argmaxLastPosition(const blackforge::runtime::Tensor& logits) {
    std::size_t vocabSize = logits.dim(2);
    std::size_t lastPos = logits.dim(1) - 1;

    std::size_t bestId = 0;
    float bestVal = logits.at(lastPos * vocabSize);
    for (std::size_t v = 1; v < vocabSize; ++v) {
        float val = logits.at(lastPos * vocabSize + v);
        if (val > bestVal) {
            bestVal = val;
            bestId = v;
        }
    }
    return static_cast<std::uint32_t>(bestId);
}

// Genera testo autoregressivamente da un modello linguistico allenato,
// usando la cache K/V (Model::forwardIncremental) su CPU o su GPU (vedi
// backend::cpu::Model::forwardIncremental / backend::cuda::Model::
// forwardIncremental, stessa semantica su entrambi i backend).
int runGenerate(const std::string& path, const std::string& fromCheckpoint, const std::string& tokenizerPath,
                 const std::string& prompt, std::size_t maxNewTokens, const std::string& device) {
    if (fromCheckpoint.empty()) {
        std::cerr << "errore: comando 'generate' richiede '--from-checkpoint' (generare da pesi casuali non ha "
                     "senso)\n";
        return 2;
    }
    if (tokenizerPath.empty()) {
        std::cerr << "errore: comando 'generate' richiede '--tokenizer <file.bftok>'\n";
        return 2;
    }

    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile generare, l'analisi ha trovato errori\n";
        return 1;
    }
    if (result.module.models.empty()) {
        std::cerr << "errore: il programma non definisce alcun modello da usare per la generazione\n";
        return 1;
    }
    const blackforge::ir::ModelIR& model = result.module.models.front();

    try {
        blackforge::tokenizer::Tokenizer tok = blackforge::tokenizer::loadTokenizer(tokenizerPath);

        std::vector<std::uint32_t> promptIds = tok.encode(prompt);
        if (promptIds.empty()) {
            std::cerr << "errore: il prompt codificato e' vuoto (prompt vuoto?)\n";
            return 1;
        }
        std::vector<float> promptFloats;
        promptFloats.reserve(promptIds.size());
        for (const std::uint32_t id : promptIds) promptFloats.push_back(static_cast<float>(id));
        blackforge::runtime::Tensor promptTensor({1, promptIds.size()}, promptFloats);

        std::vector<std::uint32_t> generated = promptIds;

        if (!spec.isCuda) {
            blackforge::backend::cpu::Model loadedModel(model);
            blackforge::backend::cpu::loadCheckpoint(loadedModel, fromCheckpoint);
            loadedModel.resetGenerationState();

            blackforge::runtime::Tensor logits = loadedModel.forwardIncremental(promptTensor);
            for (std::size_t step = 0; step < maxNewTokens; ++step) {
                std::uint32_t bestId = argmaxLastPosition(logits);
                generated.push_back(bestId);

                try {
                    blackforge::runtime::Tensor nextInput({1, 1}, {static_cast<float>(bestId)});
                    logits = loadedModel.forwardIncremental(nextInput);
                } catch (const std::exception&) {
                    std::cout << "(lunghezza massima di sequenza del modello raggiunta, generazione interrotta)\n";
                    break;
                }
            }
        } else {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);
            blackforge::backend::cuda::Model loadedModel(model);
            blackforge::backend::cuda::loadCheckpoint(loadedModel, fromCheckpoint);
            loadedModel.resetGenerationState();

            blackforge::runtime::Tensor logits =
                loadedModel.forwardIncremental(blackforge::backend::cuda::DeviceTensor::fromHost(promptTensor))
                    .toHost();
            for (std::size_t step = 0; step < maxNewTokens; ++step) {
                std::uint32_t bestId = argmaxLastPosition(logits);
                generated.push_back(bestId);

                try {
                    blackforge::runtime::Tensor nextInput({1, 1}, {static_cast<float>(bestId)});
                    logits = loadedModel
                                 .forwardIncremental(blackforge::backend::cuda::DeviceTensor::fromHost(nextInput))
                                 .toHost();
                } catch (const std::exception&) {
                    std::cout << "(lunghezza massima di sequenza del modello raggiunta, generazione interrotta)\n";
                    break;
                }
            }
#endif
        }

        std::cout << tok.decode(generated) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di generazione: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runTokenizerTrain(const std::string& corpusPath, std::size_t vocabSize, const std::string& outputPath) {
    std::string corpus;
    if (!readFile(corpusPath, corpus)) {
        std::cerr << "errore: impossibile leggere il corpus '" << corpusPath << "'\n";
        return 2;
    }
    if (vocabSize == 0) {
        std::cerr << "errore: '--vocab-size' richiede un intero positivo (deve superare "
                   << blackforge::tokenizer::Tokenizer::kFirstMergeId << ", i token base + speciali)\n";
        return 2;
    }
    if (outputPath.empty()) {
        std::cerr << "errore: comando 'tokenizer-train' richiede '--output <file.bftok>'\n";
        return 2;
    }

    try {
        blackforge::tokenizer::Tokenizer tok;
        tok.train(corpus, vocabSize);
        blackforge::tokenizer::saveTokenizer(tok, outputPath);
        std::cout << "Tokenizer addestrato: " << tok.vocabSize() << " token totali (" << tok.merges().size()
                   << " merge appresi da un corpus di " << corpus.size() << " byte)\n";
        std::cout << "Salvato in '" << outputPath << "'\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di training del tokenizer: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runTokenizerEncode(const std::string& tokenizerPath, const std::string& textPath) {
    std::string text;
    if (!readFile(textPath, text)) {
        std::cerr << "errore: impossibile leggere il file di testo '" << textPath << "'\n";
        return 2;
    }

    try {
        blackforge::tokenizer::Tokenizer tok = blackforge::tokenizer::loadTokenizer(tokenizerPath);
        std::vector<std::uint32_t> ids = tok.encode(text);

        std::cout << ids.size() << " token (da " << text.size() << " byte, vocabolario " << tok.vocabSize()
                   << "):\n";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i] << (i + 1 < ids.size() ? " " : "\n");
        }

        std::string roundTrip = tok.decode(ids);
        if (roundTrip != text) {
            std::cerr << "attenzione: decode(encode(testo)) non coincide con il testo originale (non dovrebbe mai "
                         "accadere per un tokenizer byte-level corretto: possibile bug)\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "errore di codifica: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Legge il corpus a pezzi e lo codifica man mano, invece di caricarlo tutto
// in memoria.
//
// readFile() accumula in un ostringstream e poi copia il risultato in una
// stringa: il picco e' il doppio della dimensione del file. Su un corpus da
// 11,5 GB sono 23 GB, e quando la memoria non basta ostringstream smette di
// accumulare SENZA lanciare eccezioni: si ottiene un corpus troncato e
// nessun errore. Un addestramento lanciato su quel dataset userebbe una
// frazione del testo senza che nulla lo segnali.
//
// I pezzi sono tagliati sull'ultimo a capo, mai a meta' riga: la
// pre-tokenizzazione lavora per parole, quindi spezzare fra due righe non
// cambia il risultato rispetto alla codifica del testo intero.
bool encodeCorpusStreaming(const std::string& path, const blackforge::tokenizer::Tokenizer& tok,
                            std::vector<std::uint32_t>& ids) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        return false;
    }
    constexpr std::size_t kChunkBytes = 64ULL * 1024ULL * 1024ULL;
    std::string buffer;
    std::string chunk(kChunkBytes, '\0');
    while (file) {
        file.read(chunk.data(), static_cast<std::streamsize>(kChunkBytes));
        const std::streamsize got = file.gcount();
        if (got <= 0) {
            break;
        }
        buffer.append(chunk.data(), static_cast<std::size_t>(got));
        const std::size_t cut = buffer.find_last_of('\n');
        if (cut == std::string::npos) {
            continue;  // nessuna riga completa: si accumula ancora
        }
        const std::string head = buffer.substr(0, cut + 1);
        buffer.erase(0, cut + 1);
        const std::vector<std::uint32_t> part = tok.encode(head);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    if (!buffer.empty()) {
        const std::vector<std::uint32_t> part = tok.encode(buffer);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return true;
}

int runDatasetBuild(const std::string& corpusPath, const std::string& tokenizerPath, std::size_t seqLen,
                     const std::string& outputPath, bool mlm, float maskProb) {
    if (seqLen == 0) {
        std::cerr << "errore: '--seq-len' richiede un intero positivo\n";
        return 2;
    }
    if (outputPath.empty()) {
        std::cerr << "errore: comando 'dataset-build' richiede '--output <file.bfdata>'\n";
        return 2;
    }

    try {
        blackforge::tokenizer::Tokenizer tok = blackforge::tokenizer::loadTokenizer(tokenizerPath);
        std::vector<std::uint32_t> ids;
        if (!encodeCorpusStreaming(corpusPath, tok, ids)) {
            std::cerr << "errore: impossibile leggere il corpus '" << corpusPath << "'\n";
            return 2;
        }
        std::cout << "Corpus codificato: " << ids.size() << " token\n";
        std::size_t numExamples;
        if (mlm) {
            std::string corpus;
            if (!readFile(corpusPath, corpus)) {
                std::cerr << "errore: impossibile leggere il corpus\n";
                return 2;
            }
            numExamples = blackforge::tokenizer::buildMaskedLanguageModelDataset(tok, corpus, seqLen, maskProb,
                                                                                   /*seed=*/42, outputPath);
            std::cout << "Dataset MLM costruito: " << numExamples << " esempi (seqLen=" << seqLen
                       << ", maskProb=" << maskProb << ", vocabolario=" << tok.vocabSize()
                       << "). Allena con 'loss cross_entropy_masked' e 'bidirectional_attention' nella "
                          "pipeline.\n";
        } else {
            numExamples = blackforge::tokenizer::buildLanguageModelDatasetFromIds(ids, seqLen, outputPath);
            std::cout << "Dataset costruito: " << numExamples << " esempi (seqLen=" << seqLen
                       << ", vocabolario=" << tok.vocabSize() << ")\n";
        }
        std::cout << "Salvato in '" << outputPath << "'\n";
    } catch (const std::exception& e) {
        std::cerr << "errore di costruzione del dataset: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

std::string formatShape(const std::vector<std::size_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

// 'blackforge benchmark blackbit': misura un modello BlackBit (vedi
// blackforge/blackbit/benchmark.hpp). E' un percorso separato da
// runBenchmark() perche' BlackBit non e' descritto nel linguaggio: non
// c'e' nessun file .bf da leggere, solo una configurazione JSON o il
// nome di un preset.
int runBlackBitBenchmark(const std::string& configPath, const std::string& presetName, std::size_t seqLen,
                          std::size_t microBatch, std::size_t steps, std::size_t warmupSteps,
                          std::size_t maxVramMb, std::size_t optimizerRank, const std::string& recomputeMode,
                          bool dryRun, bool instantiateOnly, bool seqLadder, bool profile,
                          const std::string& device) {
    try {
        blackforge::blackbit::BlackBitConfig config =
            configPath.empty() ? blackforge::blackbit::blackBitPreset(presetName.empty() ? "tiny" : presetName)
                                : blackforge::blackbit::loadConfigFromJson(configPath);

        blackforge::blackbit::BenchmarkOptions options;
        if (seqLen != 0) {
            options.seqLen = seqLen;
        }
        if (microBatch != 0) {
            options.microBatch = microBatch;
        }
        if (steps != 0) {
            options.steps = steps;
        }
        if (instantiateOnly) {
            options.steps = 0;
            options.warmupSteps = 0;
        }
        options.warmupSteps = warmupSteps;
        options.maxVramMb = maxVramMb;
        options.dryRun = dryRun;
        if (optimizerRank != 0) {
            options.optimizer.rank = optimizerRank;
        }

        if (recomputeMode == "none") {
            options.runtime.recompute = blackforge::blackbit::ActivationRecompute::None;
        } else if (recomputeMode == "per-layer" || recomputeMode.empty()) {
            options.runtime.recompute = blackforge::blackbit::ActivationRecompute::PerLayer;
        } else if (recomputeMode == "full") {
            options.runtime.recompute = blackforge::blackbit::ActivationRecompute::FullRecompute;
        } else if (recomputeMode.rfind("every-", 0) == 0) {
            options.runtime.recompute = blackforge::blackbit::ActivationRecompute::EveryNLayers;
            options.runtime.recomputeEveryN =
                static_cast<std::size_t>(std::strtoull(recomputeMode.substr(6).c_str(), nullptr, 10));
            if (options.runtime.recomputeEveryN == 0) {
                std::cerr << "errore: '--recompute every-N' richiede un intero positivo\n";
                return 2;
            }
        } else {
            std::cerr << "errore: '--recompute' accetta none, per-layer, every-N o full\n";
            return 2;
        }

        DeviceSpec spec;
        std::string deviceError;
        if (!parseDeviceSpec(device, spec, deviceError)) {
            std::cerr << "errore: " << deviceError << "\n";
            return 2;
        }
#if BLACKFORGE_HAS_CUDA
        // Il profiler costa una cudaEventRecord per regione e una sola
        // sincronizzazione a fine passo, ma resta spento di default:
        // il benchmark deve misurare il percorso di produzione.
        blackforge::blackbit::cuda::GpuProfiler::instance().setEnabled(profile && spec.isCuda && !dryRun);
#else
        if (profile) {
            std::cerr << "errore: '--profile' richiede una build CUDA\n";
            return 2;
        }
#endif
        if (!spec.isCuda || dryRun) {
            const blackforge::blackbit::BenchmarkResult result =
                blackforge::blackbit::runBlackBitBenchmark(config, options, dryRun ? nullptr : &std::cout);
            std::cout << "\n" << result.report();
            return result.withinBudget ? 0 : 1;
        }
#if BLACKFORGE_HAS_CUDA
        if (seqLadder) {
            std::vector<std::size_t> lengths;
            for (const std::size_t length : {8U, 16U, 32U, 64U, 128U, 256U, 512U}) {
                if (length <= config.maxSeqLen) lengths.push_back(length);
            }
            const auto results = blackforge::blackbit::cuda::runBenchmarkLadder(
                config, options, lengths, spec.cudaIndex, &std::cout);
            return std::all_of(results.begin(), results.end(), [](const auto& result) {
                return result.withinBudget && result.nanInfCount == 0 && result.ternaryFlips > 0;
            }) ? 0 : 1;
        }
        const blackforge::blackbit::cuda::BenchmarkResult result =
            blackforge::blackbit::cuda::runBenchmark(config, options, spec.cudaIndex, &std::cout);
        std::cout << "\n" << result.report();
        return result.withinBudget ? 0 : 1;
#else
        std::cerr << "errore: benchmark BlackBit CUDA richiesto ma questa build non include CUDA\n";
        return 1;
#endif
    } catch (const std::exception& error) {
        std::cerr << "errore: " << error.what() << "\n";
        return 1;
    }
}


// Genera testo da un checkpoint BlackBit.
//
// Il motore di addestramento sapeva solo addestrare: non esisteva alcun modo
// di far produrre una parola a un modello BlackBit, e ogni giudizio sulla sua
// qualita' passava dalla loss, che e' un indicatore indiretto. Un modello puo'
// avere una loss decorosa e generare ripetizioni inutilizzabili.
//
// Nessuna cache delle chiavi e dei valori: a ogni token si rifa' il forward
// sull'intera finestra. E' uno spreco irrilevante su un modello da 60 M e
// risparmia la parte piu' delicata da scrivere. Se servira' velocita' vera si
// aggiunge dopo.
int runBlackBitEval(const std::string& checkpointPath, const std::string& datasetPath,
                     std::size_t microBatch, std::size_t maxExamples, double bytesPerToken,
                     std::size_t maxVramMb, const std::string& device) {
#if BLACKFORGE_HAS_CUDA
    try {
        DeviceSpec spec;
        std::string deviceError;
        if (!parseDeviceSpec(device, spec, deviceError) || !spec.isCuda) {
            std::cerr << "errore: 'blackbit-eval' richiede --device cuda o cuda:N\n";
            return 2;
        }
        blackforge::blackbit::cuda::EvalOptions options;
        options.checkpoint = checkpointPath;
        options.datasetPath = datasetPath;
        options.microBatch = microBatch == 0 ? 16 : microBatch;
        options.maxExamples = maxExamples;
        options.bytesPerToken = bytesPerToken;
        options.maxVramMb = maxVramMb;

        const auto r = blackforge::blackbit::cuda::evaluate(options, spec.cudaIndex, &std::cerr);
        std::cerr << "                                        \r";

        std::cout << std::fixed << std::setprecision(4)
                  << "passo " << r.checkpointStep
                  << "  token_visti " << r.checkpointTokens
                  << "  loss " << r.loss
                  << "  ppl " << std::setprecision(2) << r.perplexity;
        if (r.bitsPerByte > 0.0) std::cout << "  bpb " << std::setprecision(4) << r.bitsPerByte;
        std::cout << "  esempi " << r.examples
                  << "  token " << r.scoredTokens
                  << "  secondi " << std::setprecision(1) << r.seconds << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "errore: " << e.what() << "\n";
        return 1;
    }
#else
    (void)checkpointPath; (void)datasetPath; (void)microBatch; (void)maxExamples;
    (void)bytesPerToken; (void)maxVramMb; (void)device;
    std::cerr << "errore: questa build non include il supporto CUDA\n";
    return 2;
#endif
}

int runBlackBitGenerate(const std::string& checkpointPath, const std::string& tokenizerPath,
                         const std::string& prompt, std::size_t maxNewTokens, float temperature,
                         float topP, std::size_t topK, float repetitionPenalty,
                         std::size_t penaltyWindow, unsigned int seed) {
    try {
        const blackforge::blackbit::BlackBitConfig config =
            blackforge::blackbit::readCheckpointConfig(checkpointPath);
        blackforge::blackbit::BlackBitModel model(config);
        const blackforge::blackbit::BlackBitTrainingState state =
            blackforge::blackbit::loadCheckpoint(checkpointPath, model);
        const blackforge::tokenizer::Tokenizer tok = blackforge::tokenizer::loadTokenizer(tokenizerPath);

        std::cerr << "checkpoint al passo " << state.step << ", " << state.tokensSeen
                   << " token visti\n";
        std::cerr << "campionamento: temperatura " << temperature << ", top-p " << topP
                   << ", top-k " << topK << ", penalita ripetizione " << repetitionPenalty
                   << " su finestra " << penaltyWindow << "\n";

        const std::vector<std::uint32_t> encoded = tok.encode(prompt);
        std::vector<int> tokens;
        tokens.reserve(encoded.size() + maxNewTokens);
        for (std::uint32_t id : encoded) tokens.push_back(static_cast<int>(id));
        if (tokens.empty()) {
            std::cerr << "errore: il prompt e' vuoto dopo la tokenizzazione\n";
            return 2;
        }

        std::mt19937 rng(seed);
        const std::size_t generati = maxNewTokens;
        for (std::size_t n = 0; n < generati; ++n) {
            const std::size_t seq = std::min(tokens.size(), config.maxSeqLen);
            const std::vector<int> finestra(tokens.end() - static_cast<std::ptrdiff_t>(seq), tokens.end());
            blackforge::blackbit::BlackBitForwardCache cache;
            const blackforge::runtime::Tensor hidden = model.forwardHidden(finestra, 1, seq, cache);
            const blackforge::runtime::Tensor logits = model.logits(hidden);
            const std::size_t vocab = config.vocabSize;
            const float* riga = logits.data().data() + (seq - 1) * vocab;

            // Penalita' di ripetizione sui logit grezzi, prima di tutto il resto.
            //
            // I cicli di ripetizione non sono un sintomo di sottoaddestramento:
            // anche modelli grandi e completi degenerano se campionati
            // massimizzando la verosimiglianza senza correttivi. Serve quindi
            // anche - e soprattutto - a temperatura 0, dove la decodifica greedy
            // nuda cade in "......" dopo pochi token.
            //
            // Forma di Keskar e colleghi: si divide il logit positivo e si
            // moltiplica quello negativo, cosi' la penalita' allontana sempre
            // dal token gia' visto qualunque sia il segno.
            std::vector<float> logit(riga, riga + vocab);
            if (repetitionPenalty > 1.0F && !tokens.empty()) {
                const std::size_t quanti = penaltyWindow == 0 ? tokens.size()
                                                              : std::min(penaltyWindow, tokens.size());
                for (std::size_t k = tokens.size() - quanti; k < tokens.size(); ++k) {
                    const std::size_t id = static_cast<std::size_t>(tokens[k]);
                    if (id >= vocab) continue;
                    logit[id] = logit[id] > 0.0F ? logit[id] / repetitionPenalty
                                                 : logit[id] * repetitionPenalty;
                }
            }

            int scelto = 0;
            if (temperature <= 0.0F) {
                float migliore = logit[0];
                for (std::size_t v = 1; v < vocab; ++v) {
                    if (logit[v] > migliore) { migliore = logit[v]; scelto = static_cast<int>(v); }
                }
            } else {
                // softmax con temperatura, stabilizzato sottraendo il massimo
                float massimo = logit[0];
                for (std::size_t v = 1; v < vocab; ++v) massimo = std::max(massimo, logit[v]);
                std::vector<double> pesi(vocab);
                double somma = 0.0;
                for (std::size_t v = 0; v < vocab; ++v) {
                    pesi[v] = std::exp(static_cast<double>((logit[v] - massimo) / temperature));
                    somma += pesi[v];
                }

                // Ordino per probabilita' decrescente per poter tagliare la coda.
                std::vector<std::size_t> ordine(vocab);
                for (std::size_t v = 0; v < vocab; ++v) ordine[v] = v;
                std::sort(ordine.begin(), ordine.end(),
                          [&pesi](std::size_t a, std::size_t b) { return pesi[a] > pesi[b]; });

                // top-k: si tengono i k piu' probabili.
                std::size_t tenuti = topK == 0 ? vocab : std::min(topK, vocab);

                // top-p (nucleus): si tiene il minimo insieme la cui massa
                // cumulata raggiunge p. Taglia la coda lunga di token quasi
                // impossibili che, sommati su un vocabolario da migliaia di
                // voci, vengono estratti abbastanza spesso da rovinare il testo.
                if (topP > 0.0F && topP < 1.0F) {
                    double cumulata = 0.0;
                    std::size_t nucleo = 0;
                    for (; nucleo < tenuti; ++nucleo) {
                        cumulata += pesi[ordine[nucleo]] / somma;
                        if (cumulata >= static_cast<double>(topP)) { ++nucleo; break; }
                    }
                    tenuti = std::max<std::size_t>(1, std::min(tenuti, nucleo));
                }

                double massa = 0.0;
                for (std::size_t i = 0; i < tenuti; ++i) massa += pesi[ordine[i]];
                std::uniform_real_distribution<double> uniforme(0.0, massa);
                double soglia = uniforme(rng);
                scelto = static_cast<int>(ordine[tenuti - 1]);
                for (std::size_t i = 0; i < tenuti; ++i) {
                    soglia -= pesi[ordine[i]];
                    if (soglia <= 0.0) { scelto = static_cast<int>(ordine[i]); break; }
                }
            }
            tokens.push_back(scelto);
        }

        std::vector<std::uint32_t> uscita;
        uscita.reserve(tokens.size());
        for (int id : tokens) uscita.push_back(static_cast<std::uint32_t>(id));
        std::cout << tok.decode(uscita) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "errore: " << e.what() << "\n";
        return 1;
    }
}

int runBlackBitTraining(const std::string& configPath, const std::string& presetName,
                        const std::string& datasetPath, const std::string& tokenizerPath,
                        std::size_t microBatch, std::size_t steps,
                        std::size_t maxVramMb, std::size_t optimizerRank,
                        const std::string& recomputeMode, const std::string& fromCheckpoint,
                        const std::string& saveCheckpoint, float learningRate,
                        std::size_t warmupSteps, std::size_t scheduleSteps,
                        std::size_t checkpointEvery, bool errorFeedback, float latentClamp, float latentDecay,
                        const std::string& device) {
#if BLACKFORGE_HAS_CUDA
    try {
        DeviceSpec spec;
        std::string deviceError;
        if (!parseDeviceSpec(device, spec, deviceError) || !spec.isCuda) {
            std::cerr << "errore: 'train blackbit' richiede --device cuda o cuda:N\n";
            return 2;
        }
        blackforge::blackbit::BlackBitConfig config =
            configPath.empty() ? blackforge::blackbit::blackBitPreset(presetName.empty() ? "tiny" : presetName)
                               : blackforge::blackbit::loadConfigFromJson(configPath);
        blackforge::blackbit::cuda::TrainingOptions options;
        options.datasetPath = datasetPath;
        options.tokenizerPath = tokenizerPath;
        options.fromCheckpoint = fromCheckpoint;
        options.saveCheckpoint = saveCheckpoint;
        options.microBatch = microBatch == 0 ? 1 : microBatch;
        options.steps = steps == 0 ? 100 : steps;
        options.maxVramMb = maxVramMb;
        if (optimizerRank != 0) options.optimizer.rank = optimizerRank;
        if (learningRate > 0.0F) options.optimizer.learningRate = learningRate;
        options.warmupSteps = warmupSteps;
        options.scheduleSteps = scheduleSteps;
        options.checkpointEvery = checkpointEvery;
        options.optimizer.errorFeedback = errorFeedback;
        if (latentClamp > 0.0F) options.optimizer.latentClamp = latentClamp;
        if (latentDecay > 0.0F) options.optimizer.latentDecay = latentDecay;
        if (recomputeMode == "none") {
            options.runtime.recompute = blackforge::blackbit::ActivationRecompute::None;
        } else if (recomputeMode.empty() || recomputeMode == "per-layer") {
            options.runtime.recompute = blackforge::blackbit::ActivationRecompute::PerLayer;
        } else {
            std::cerr << "errore: il trainer CUDA corrente supporta --recompute none o per-layer\n";
            return 2;
        }
        const auto result = blackforge::blackbit::cuda::train(config, options, spec.cudaIndex, &std::cout);
        std::cout << "\n" << result.report();
        return result.nanInfCount == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "errore: " << error.what() << "\n";
        return 1;
    }
#else
    (void)configPath; (void)presetName; (void)datasetPath; (void)tokenizerPath; (void)microBatch; (void)steps;
    (void)maxVramMb; (void)optimizerRank; (void)recomputeMode; (void)fromCheckpoint;
    (void)saveCheckpoint; (void)device;
    std::cerr << "errore: 'train blackbit' richiede una build CUDA\n";
    return 1;
#endif
}

int runBenchmark(const std::string& path, const std::string& device, std::size_t batchSize,
                  std::size_t warmupIterations, std::size_t measuredIterations) {
    DeviceSpec spec;
    std::string deviceError;
    if (!parseDeviceSpec(device, spec, deviceError)) {
        std::cerr << "errore: " << deviceError << "\n";
        return 2;
    }
    if (spec.isCuda && !BLACKFORGE_HAS_CUDA) {
        std::cerr << "errore: '--device " << device << "' richiesto ma questa build di blackforge e' stata "
                     "compilata senza supporto CUDA (BLACKFORGE_ENABLE_CUDA=OFF in fase di compilazione)\n";
        return 1;
    }

    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile eseguire il benchmark, l'analisi ha trovato errori\n";
        return 1;
    }
    if (result.module.models.empty()) {
        std::cerr << "errore: il programma non definisce alcun modello da misurare\n";
        return 1;
    }

    const blackforge::ir::ModelIR& model = result.module.models.front();
    std::string dtypeName = blackforge::sema::dtypeName(model.valueById(model.inputValue).dtype);

    try {
        blackforge::backend::cpu::BenchmarkResult cpuResult = blackforge::backend::cpu::runBenchmark(
            model, batchSize, warmupIterations, measuredIterations, result.module.precision);

        std::cout << "Benchmark di '" << model.name << "'\n";
        std::cout << "  hardware:          CPU (backend di riferimento; nome del processore non rilevato)\n";
        if (result.module.precision.has_value()) {
            std::cout << "  precisione:        " << dtypeName << " dichiarata; quantizzazione simulata applicata "
                       << "(storage=" << blackforge::sema::dtypeName(result.module.precision->storage)
                       << " compute=" << blackforge::sema::dtypeName(result.module.precision->compute)
                       << "), accumulo interno sempre in float32\n";
        } else {
            std::cout << "  precisione:        " << dtypeName
                       << " dichiarata (nessun blocco 'precision': calcolo in piena float32)\n";
        }
        std::cout << "  forma input:       " << formatShape(cpuResult.inputShape) << "\n";
        std::cout << "  warmup:            " << cpuResult.warmupIterations << " iterazioni\n";
        std::cout << "  iterazioni:        " << cpuResult.measuredIterations << "\n";
        std::cout << "  tempo medio:       " << cpuResult.meanMilliseconds << " ms\n";
        std::cout << "  throughput:        " << cpuResult.throughputSamplesPerSecond << " esempi/s\n";
        std::cout << "  memoria (stimata): " << (static_cast<double>(cpuResult.estimatedMemoryBytes) / 1024.0 / 1024.0)
                   << " MiB (float32; e' una stima teorica, non memoria di processo misurata)\n";
        std::cout << "  per operazione (CPU, ciascuna misurata separatamente sul proprio input reale):\n";
        for (const auto& op : cpuResult.perOperation) {
            std::cout << "    [" << op.operationIndex << "] " << op.operationName << ": " << op.meanMilliseconds
                       << " ms\n";
        }

        if (spec.isCuda) {
#if BLACKFORGE_HAS_CUDA
            blackforge::backend::cuda::setActiveDevice(spec.cudaIndex);

            std::string gpuName = "GPU sconosciuta";
            for (const auto& d : blackforge::backend::cuda::enumerateDevices()) {
                if (d.index == spec.cudaIndex) {
                    gpuName = d.name;
                }
            }

            blackforge::backend::cuda::Executor cudaExecutor;
            blackforge::runtime::Tensor cudaInput =
                cudaExecutor.makeSyntheticInput(model, batchSize);

            for (std::size_t i = 0; i < warmupIterations; ++i) {
                blackforge::runtime::Tensor warm = cudaExecutor.run(model, cudaInput);
                (void)warm;
            }

            blackforge::runtime::Tensor gpuOutput;
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < measuredIterations; ++i) {
                gpuOutput = cudaExecutor.run(model, cudaInput);
            }
            auto end = std::chrono::steady_clock::now();

            double totalMillis = std::chrono::duration<double, std::milli>(end - start).count();
            double gpuMeanMillis = totalMillis / static_cast<double>(measuredIterations);
            double gpuThroughput = gpuMeanMillis > 0.0 ? (static_cast<double>(batchSize) * 1000.0) / gpuMeanMillis
                                                         : 0.0;

            std::cout << "\n  hardware GPU:      " << gpuName << " (cuda:" << spec.cudaIndex << ")\n";
            std::cout << "  tempo medio GPU:   " << gpuMeanMillis << " ms\n";
            std::cout << "  throughput GPU:    " << gpuThroughput << " esempi/s\n";
            std::cout << "  speedup vs CPU:    " << (cpuResult.meanMilliseconds / gpuMeanMillis) << "x\n";

            // Confronto con la modalita' di riferimento (CPU): stesso
            // seme, quindi stesso input e stessi pesi iniziali.
            blackforge::backend::cpu::Executor cpuExecutor;
            blackforge::runtime::Tensor cpuInput =
                cpuExecutor.makeSyntheticInput(model, batchSize);
            blackforge::runtime::Tensor cpuOutput = cpuExecutor.run(model, cpuInput);

            float maxAbsDiff = 0.0F;
            for (std::size_t i = 0; i < cpuOutput.elementCount() && i < gpuOutput.elementCount(); ++i) {
                maxAbsDiff = std::max(maxAbsDiff, std::fabs(cpuOutput.at(i) - gpuOutput.at(i)));
            }
            std::cout << "  scarto massimo vs CPU (riferimento): " << maxAbsDiff << "\n";
#endif
        }
    } catch (const std::exception& e) {
        std::cerr << "errore di benchmark: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int runInspect(const std::string& path) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": impossibile ispezionare, l'analisi ha trovato errori\n";
        return 1;
    }

    std::cout << path << "\n";
    if (result.module.target.has_value()) {
        std::cout << "target: " << *result.module.target << "\n";
    }
    if (result.module.precision.has_value()) {
        std::cout << "precision: storage=" << blackforge::sema::dtypeName(result.module.precision->storage)
                   << " compute=" << blackforge::sema::dtypeName(result.module.precision->compute)
                   << " accumulate=" << blackforge::sema::dtypeName(result.module.precision->accumulate)
                   << " (applicata da 'run'/'forecast'/'benchmark' su CPU; non ancora su CUDA; ignorata durante "
                      "'train')\n";
    }
    std::cout << result.module.models.size() << " modello/i\n";

    for (const auto& model : result.module.models) {
        std::cout << "\nmodello '" << model.name << "'\n";
        std::cout << "  input: " << blackforge::ir::typeString(model.valueById(model.inputValue)) << "\n";

        try {
            blackforge::backend::cpu::Model built(model);
            std::size_t totalParams = 0;
            for (blackforge::backend::cpu::Parameter* param : built.allParameters()) {
                totalParams += param->value.elementCount();
            }
            std::cout << "  parametri: " << totalParams << " ("
                       << (static_cast<double>(totalParams * sizeof(float)) / 1024.0 / 1024.0)
                       << " MiB come float32)\n";
        } catch (const std::exception& e) {
            std::cout << "  parametri: non calcolabili (" << e.what() << ")\n";
        }

        for (const auto& pipeline : model.pipelines) {
            std::cout << "  pipeline:\n";
            for (const auto& op : pipeline.operations) {
                std::cout << "    " << blackforge::ir::opKindName(op.kind);
                if (op.kind == blackforge::ir::OpKind::Linear) {
                    std::cout << "(" << op.linearOutFeatures << ")";
                }
                std::cout << " -> " << blackforge::ir::typeString(model.valueById(op.output)) << "\n";
            }
        }
    }

    return 0;
}

int runBuild(const std::string& path) {
    CompileOutput result = compile(path, /*verbose=*/false, /*printAst=*/false, /*printIr=*/false);
    if (result.status == CompileStatus::FileNotFound) {
        return 2;
    }
    if (result.status == CompileStatus::AnalysisFailed) {
        std::cerr << path << ": build fallita, l'analisi ha trovato errori\n";
        return 1;
    }

    // A differenza di 'check' (che valida solo AST/IR), 'build' prova
    // davvero a costruire ogni modello (allocando i suoi parametri):
    // rileva errori che richiedono la costruzione effettiva, come una
    // dimensione delle feature ancora simbolica (impossibile allocare
    // pesi concreti).
    bool ok = true;
    for (const auto& model : result.module.models) {
        try {
            blackforge::backend::cpu::Model built(model);
            std::size_t totalParams = 0;
            for (blackforge::backend::cpu::Parameter* param : built.allParameters()) {
                totalParams += param->value.elementCount();
            }
            std::cout << "  modello '" << model.name << "': OK (" << totalParams << " parametri)\n";
        } catch (const std::exception& e) {
            std::cerr << "  modello '" << model.name << "': errore di costruzione: " << e.what() << "\n";
            ok = false;
        }
    }

    if (!ok) {
        std::cerr << path << ": build fallita\n";
        return 1;
    }

    std::cout << path << ": build riuscita (" << result.module.models.size()
               << " modello/i pronti per l'esecuzione)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.empty()) {
        printUsage();
        return 2;
    }

    bool verbose = false;
    bool printAst = false;
    bool printIr = false;
    std::size_t batchSize = 1;
    std::string device = "cpu";
    std::string devicesList;
    std::string fromCheckpoint;
    std::string saveCheckpointPath;
    std::size_t warmupIterations = 5;
    bool warmupGiven = false;
    std::size_t measuredIterations = 20;
    std::size_t vocabSize = 0;
    std::size_t seqLen = 0;
    std::string outputPath;
    std::string tokenizerPath;
    std::string prompt;
    std::size_t maxNewTokens = 50;
    bool mlm = false;
    float maskProb = 0.15F;
    std::string configPath;
    std::string datasetPath;
    std::size_t microBatch = 0;
    std::size_t steps = 0;
    std::size_t maxVramMb = 7800;
    std::size_t optimizerRank = 0;
    float learningRate = 0.0F;
    std::size_t warmupSteps = 0;
    std::size_t scheduleSteps = 0;
    std::size_t checkpointEvery = 0;
    bool errorFeedback = false;
    float latentClamp = 0.0F;
    float latentDecay = 0.0F;
    double bytesPerToken = 0.0;
    float topP = 0.9F;
    std::size_t topK = 0;
    float repetitionPenalty = 1.1F;
    std::size_t penaltyWindow = 64;
    std::size_t maxExamples = 0;
    float temperature = 0.0F;
    std::string recomputeMode;
    bool dryRun = false;
    bool instantiateOnly = false;
    bool seqLadder = false;
    bool profile = false;
    std::vector<std::string> positional;
    std::string command = args.front();

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--verbose") {
            verbose = true;
        } else if (args[i] == "--print-ast") {
            printAst = true;
        } else if (args[i] == "--print-ir") {
            printIr = true;
        } else if (args[i] == "--batch") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--batch' richiede un valore\n";
                return 2;
            }
            batchSize = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
            if (batchSize == 0) {
                std::cerr << "errore: '--batch' richiede un intero positivo\n";
                return 2;
            }
        } else if (args[i] == "--device") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--device' richiede un valore (cpu o cuda)\n";
                return 2;
            }
            device = args[++i];
        } else if (args[i] == "--devices") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--devices' richiede un elenco di indici GPU separati da virgola (es. "
                             "'0,1')\n";
                return 2;
            }
            devicesList = args[++i];
        } else if (args[i] == "--from-checkpoint") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--from-checkpoint' richiede un percorso\n";
                return 2;
            }
            fromCheckpoint = args[++i];
        } else if (args[i] == "--save-checkpoint") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--save-checkpoint' richiede un percorso\n";
                return 2;
            }
            saveCheckpointPath = args[++i];
        } else if (args[i] == "--warmup") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--warmup' richiede un valore\n";
                return 2;
            }
            warmupIterations = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
            warmupGiven = true;
        } else if (args[i] == "--iterations") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--iterations' richiede un valore\n";
                return 2;
            }
            measuredIterations = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
            if (measuredIterations == 0) {
                std::cerr << "errore: '--iterations' richiede un intero positivo\n";
                return 2;
            }
        } else if (args[i] == "--vocab-size") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--vocab-size' richiede un valore\n";
                return 2;
            }
            vocabSize = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--seq-len") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--seq-len' richiede un valore\n";
                return 2;
            }
            seqLen = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--output") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--output' richiede un percorso\n";
                return 2;
            }
            outputPath = args[++i];
        } else if (args[i] == "--tokenizer") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--tokenizer' richiede un percorso\n";
                return 2;
            }
            tokenizerPath = args[++i];
        } else if (args[i] == "--prompt") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--prompt' richiede un testo\n";
                return 2;
            }
            prompt = args[++i];
        } else if (args[i] == "--max-new-tokens") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--max-new-tokens' richiede un valore\n";
                return 2;
            }
            maxNewTokens = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--config") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--config' richiede un percorso\n";
                return 2;
            }
            configPath = args[++i];
        } else if (args[i] == "--dataset") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--dataset' richiede un percorso\n";
                return 2;
            }
            datasetPath = args[++i];
        } else if (args[i] == "--micro-batch") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--micro-batch' richiede un valore\n";
                return 2;
            }
            microBatch = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--steps") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--steps' richiede un valore\n";
                return 2;
            }
            steps = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--max-vram-mb") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--max-vram-mb' richiede un valore (0 = nessun limite)\n";
                return 2;
            }
            maxVramMb = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--temperature") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--temperature' richiede un valore\n";
                return 2;
            }
            temperature = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--top-p") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--top-p' richiede un valore\n";
                return 2;
            }
            topP = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--top-k") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--top-k' richiede un valore\n";
                return 2;
            }
            topK = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--repetition-penalty") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--repetition-penalty' richiede un valore\n";
                return 2;
            }
            repetitionPenalty = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--penalty-window") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--penalty-window' richiede un valore\n";
                return 2;
            }
            penaltyWindow = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--bytes-per-token") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--bytes-per-token' richiede un valore\n";
                return 2;
            }
            bytesPerToken = std::strtod(args[++i].c_str(), nullptr);
        } else if (args[i] == "--max-examples") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--max-examples' richiede un valore\n";
                return 2;
            }
            maxExamples = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--latent-decay") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--latent-decay' richiede un valore\n";
                return 2;
            }
            latentDecay = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--latent-clamp") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--latent-clamp' richiede un valore\n";
                return 2;
            }
            latentClamp = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--error-feedback") {
            errorFeedback = true;
        } else if (args[i] == "--checkpoint-every") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--checkpoint-every' richiede un valore\n";
                return 2;
            }
            checkpointEvery = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--learning-rate") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--learning-rate' richiede un valore\n";
                return 2;
            }
            learningRate = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--warmup-steps") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--warmup-steps' richiede un valore\n";
                return 2;
            }
            warmupSteps = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--schedule-steps") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--schedule-steps' richiede un valore\n";
                return 2;
            }
            scheduleSteps = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--optimizer-rank") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--optimizer-rank' richiede un valore\n";
                return 2;
            }
            optimizerRank = static_cast<std::size_t>(std::strtoull(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--recompute") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--recompute' richiede un valore (none, per-layer, every-N, full)\n";
                return 2;
            }
            recomputeMode = args[++i];
        } else if (args[i] == "--dry-run") {
            dryRun = true;
        } else if (args[i] == "--instantiate-only") {
            instantiateOnly = true;
        } else if (args[i] == "--seq-ladder") {
            seqLadder = true;
        } else if (args[i] == "--profile") {
            profile = true;
        } else if (args[i] == "--mlm") {
            mlm = true;
        } else if (args[i] == "--mask-prob") {
            if (i + 1 >= args.size()) {
                std::cerr << "errore: '--mask-prob' richiede un valore\n";
                return 2;
            }
            maskProb = std::strtof(args[++i].c_str(), nullptr);
        } else {
            positional.push_back(args[i]);
        }
    }

    if (command == "--help" || command == "-h") {
        printUsage();
        return 0;
    }
    if (command == "--version" || command == "-v") {
        printVersion();
        return 0;
    }
    if (command == "check") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'check' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runCheck(positional.front(), verbose, printAst, printIr);
    }
    if (command == "build") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'build' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runBuild(positional.front());
    }
    if (command == "run") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'run' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runRun(positional.front(), batchSize, device, fromCheckpoint);
    }
    if (command == "train") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'train' richiede il percorso di un file .bf\n";
            return 2;
        }
        if (positional.front() == "blackbit") {
            const std::string preset = positional.size() > 1 ? positional[1] : std::string();
            return runBlackBitTraining(configPath, preset, datasetPath, tokenizerPath, microBatch, steps, maxVramMb,
                                       optimizerRank, recomputeMode, fromCheckpoint, saveCheckpointPath, learningRate, warmupSteps,
                                       scheduleSteps, checkpointEvery, errorFeedback, latentClamp, latentDecay, device);
        }
        return runTrain(positional.front(), fromCheckpoint, saveCheckpointPath, device, devicesList);
    }
    if (command == "forecast") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'forecast' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runForecast(positional.front(), fromCheckpoint, batchSize);
    }
    if (command == "benchmark") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'benchmark' richiede il percorso di un file .bf (oppure 'blackbit')\n";
            return 2;
        }
        if (positional.front() == "blackbit") {
            const std::string preset = positional.size() > 1 ? positional[1] : std::string();
            // Il default di --warmup (5) e' pensato per il benchmark del
            // linguaggio, dove un'iterazione costa millisecondi. Un passo
            // BlackBit e' molto piu' caro: se non e' stato chiesto
            // esplicitamente, ne basta uno.
            return runBlackBitBenchmark(configPath, preset, seqLen, microBatch, steps,
                                         warmupGiven ? warmupIterations : 1,
                                         maxVramMb, optimizerRank, recomputeMode, dryRun, instantiateOnly,
                                         seqLadder, profile, device);
        }
        return runBenchmark(positional.front(), device, batchSize, warmupIterations, measuredIterations);
    }
    if (command == "inspect") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'inspect' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runInspect(positional.front());
    }
    if (command == "devices") {
        printDevices();
        return 0;
    }
    if (command == "blackbit-eval") {
        if (fromCheckpoint.empty() || datasetPath.empty()) {
            std::cerr << "errore: 'blackbit-eval' richiede --from-checkpoint e --dataset\n";
            return 2;
        }
        return runBlackBitEval(fromCheckpoint, datasetPath, microBatch, maxExamples, bytesPerToken,
                               maxVramMb, device);
    }

    if (command == "blackbit-generate") {
        if (fromCheckpoint.empty() || tokenizerPath.empty()) {
            std::cerr << "errore: 'blackbit-generate' richiede --from-checkpoint e --tokenizer\n";
            return 2;
        }
        return runBlackBitGenerate(fromCheckpoint, tokenizerPath, prompt,
                                    maxNewTokens == 0 ? 50 : maxNewTokens, temperature, topP, topK,
                                    repetitionPenalty, penaltyWindow, 42U);
    }

    if (command == "tokenizer-train") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'tokenizer-train' richiede il percorso di un corpus di testo\n";
            return 2;
        }
        return runTokenizerTrain(positional.front(), vocabSize, outputPath);
    }
    if (command == "tokenizer-encode") {
        if (positional.size() < 2) {
            std::cerr << "errore: comando 'tokenizer-encode' richiede '<tokenizer.bftok> <testo.txt>'\n";
            return 2;
        }
        return runTokenizerEncode(positional[0], positional[1]);
    }
    if (command == "dataset-build") {
        if (positional.size() < 2) {
            std::cerr << "errore: comando 'dataset-build' richiede '<corpus.txt> <tokenizer.bftok>'\n";
            return 2;
        }
        return runDatasetBuild(positional[0], positional[1], seqLen, outputPath, mlm, maskProb);
    }
    if (command == "generate") {
        if (positional.empty()) {
            std::cerr << "errore: comando 'generate' richiede il percorso di un file .bf\n";
            return 2;
        }
        return runGenerate(positional.front(), fromCheckpoint, tokenizerPath, prompt, maxNewTokens, device);
    }

    std::cerr << "errore: comando sconosciuto '" << command << "'\n";
    printUsage();
    return 2;
}
