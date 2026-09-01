#include "blackforge/blackbit/low_rank_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "blackforge/blackbit/stochastic_round.hpp"
#include "blackforge/blackbit/telemetry.hpp"
#include "blackforge/blackbit/ternary_update.hpp"

namespace blackforge::blackbit {

float projectionEntry(std::uint64_t seed, std::uint64_t epoch, std::size_t row, std::size_t component,
                       std::size_t rank) {
    // Rademacher: +-1, scalata per 1/sqrt(r) cosi' che P^T P sia
    // l'identita' in valore atteso. Un bit del generatore a contatore
    // basta: niente Box-Muller, niente trascendenti, e la stessa
    // formula e' valutabile in un kernel CUDA per thread.
    const std::uint64_t mixed = counterRandom(seed ^ (epoch * 0x9E3779B97F4A7C15ULL),
                                               static_cast<std::uint64_t>(row) * rank + component);
    const float sign = (mixed & 1ULL) != 0 ? 1.0F : -1.0F;
    return sign / std::sqrt(static_cast<float>(rank));
}

LowRankProjectedOptimizer::LowRankProjectedOptimizer(LowRankOptimizerOptions options,
                                                      TernaryConsolidationOptions consolidation)
    : options_(options), consolidation_(consolidation) {
    if (options_.rank == 0) {
        throw std::invalid_argument("LowRankProjectedOptimizer: il rango deve essere positivo");
    }
    if (options_.projectionInterval == 0) {
        throw std::invalid_argument("LowRankProjectedOptimizer: projectionInterval deve essere positivo");
    }
    if (consolidation_.enabled && consolidation_.consolidationInterval == 0) {
        throw std::invalid_argument("LowRankProjectedOptimizer: consolidationInterval deve essere positivo");
    }
}

LowRankProjectedOptimizer::~LowRankProjectedOptimizer() {
    if (accountedBytes_ != 0) {
        MemoryTelemetry::instance().recordRelease(MemoryArena::Optimizer, accountedBytes_);
    }
}

void LowRankProjectedOptimizer::accountState(std::ptrdiff_t deltaBytes) {
    if (deltaBytes > 0) {
        const auto bytes = static_cast<std::size_t>(deltaBytes);
        MemoryTelemetry::instance().recordAllocation(MemoryArena::Optimizer, bytes);
        accountedBytes_ += bytes;
    }
}

void LowRankProjectedOptimizer::registerTernary(const std::string& name, TernaryTensor& weight) {
    TernaryState& state = ternary_[name];
    state.weight = &weight;
    state.rows = weight.rows();
    state.cols = weight.rowLength();
    state.seed = options_.seed ^ parameterNameHash(name);

    auto override = rankOverrides_.find(name);
    const std::size_t requested = override != rankOverrides_.end() ? override->second : options_.rank;
    // Un rango maggiore del numero di righe non aggiunge nulla: la
    // proiezione non puo' avere rango superiore alla dimensione che
    // comprime.
    state.rank = std::min(requested, state.rows);

    const std::size_t scaleCount = state.rows * weight.groupsPerRow();
    state.scaleFirstMoment.assign(scaleCount, 0.0F);
    state.scaleSecondMoment.assign(scaleCount, 0.0F);
    state.scaleAccumulator.assign(scaleCount, 0.0F);
    // Magnitudine tipica delle scale: rende l'aggiornamento relativo invece
    // che assoluto. learningRate e' calibrato sulla griglia ternaria, e
    // applicarlo tale e quale a un parametro continuo lo muoverebbe di un
    // ordine di grandezza per passo.
    double squared = 0.0;
    for (float value : weight.scales()) squared += static_cast<double>(value) * value;
    const double rms = scaleCount == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(scaleCount));
    state.scaleRms = rms > 0.0 ? static_cast<float>(rms) : 1.0F;

    const std::size_t values = state.rank * state.cols;
    state.firstMoment.assign(values, 0.0F);
    state.secondMoment.assign(values, 0.0F);
    state.accumulator.assign(values, 0.0F);
    std::size_t bytes = 3 * values * sizeof(float);
    if (consolidation_.enabled) {
        state.residual.assign(values, 0.0F);
        bytes += values * sizeof(float);
    }
    accountState(static_cast<std::ptrdiff_t>(bytes));
}

void LowRankProjectedOptimizer::registerDense(const std::string& name, std::vector<float>& values) {
    DenseState& state = dense_[name];
    state.values = &values;
    state.firstMoment.assign(values.size(), 0.0F);
    state.secondMoment.assign(values.size(), 0.0F);
    state.accumulator.assign(values.size(), 0.0F);
    accountState(static_cast<std::ptrdiff_t>(3 * values.size() * sizeof(float)));
}

void LowRankProjectedOptimizer::setRankFor(const std::string& name, std::size_t rank) {
    if (rank == 0) {
        throw std::invalid_argument("LowRankProjectedOptimizer: il rango deve essere positivo");
    }
    rankOverrides_[name] = rank;
    auto it = ternary_.find(name);
    if (it != ternary_.end()) {
        // Gia' registrato: si ricostruisce lo stato con il nuovo rango.
        TernaryTensor& weight = *it->second.weight;
        ternary_.erase(it);
        registerTernary(name, weight);
    }
}

LowRankProjectedOptimizer::TernaryState& LowRankProjectedOptimizer::ternaryStateFor(const ParameterId& id) {
    auto it = ternary_.find(id.name);
    if (it == ternary_.end()) {
        throw std::invalid_argument("LowRankProjectedOptimizer: nessun peso ternario registrato con nome '" +
                                     id.name + "'");
    }
    if (it->second.rows != id.rows || it->second.cols != id.cols) {
        throw std::invalid_argument("LowRankProjectedOptimizer: il parametro '" + id.name +
                                     "' ha una forma diversa da quella registrata");
    }
    return it->second;
}

void LowRankProjectedOptimizer::consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow,
                                                            std::size_t rowCount, const float* block) {
    TernaryState& state = ternaryStateFor(id);
    state.touched = true;

    // R += P_blocco^T G_blocco. E' l'unica cosa che il blocco lascia
    // dietro di se': dopo questa riga il gradiente puo' sparire, ed e'
    // proprio quello che il chiamante fa.
    for (std::size_t r = 0; r < rowCount; ++r) {
        const std::size_t row = firstRow + r;
        const float* gradientRow = block + r * state.cols;
        for (std::size_t j = 0; j < state.rank; ++j) {
            const float p = projectionEntry(state.seed, state.projectionEpoch, row, j, state.rank);
            if (p == 0.0F) {
                continue;
            }
            float* target = state.accumulator.data() + j * state.cols;
            for (std::size_t c = 0; c < state.cols; ++c) {
                target[c] += p * gradientRow[c];
            }
        }
    }

    // dL/ds_g = somma sui pesi del gruppo di dL/dW_ij * trit_ij, perche'
    // W_ij = trit_ij * s_g. Somma in ordine di colonna crescente, lo stesso
    // ordine del kernel CUDA, cosi' i due percorsi restano confrontabili.
    const std::size_t groupsPerRow = state.weight->groupsPerRow();
    const std::size_t groupSize = state.weight->groupSize();
    for (std::size_t r = 0; r < rowCount; ++r) {
        const std::size_t row = firstRow + r;
        const float* gradientRow = block + r * state.cols;
        for (std::size_t g = 0; g < groupsPerRow; ++g) {
            const std::size_t firstColumn = g * groupSize;
            if (firstColumn >= state.cols) break;
            const std::size_t lastColumn = std::min(firstColumn + groupSize, state.cols);
            float partial = 0.0F;
            for (std::size_t c = firstColumn; c < lastColumn; ++c) {
                partial += gradientRow[c] * static_cast<float>(state.weight->tritAt(row * state.cols + c));
            }
            state.scaleAccumulator[row * groupsPerRow + g] += partial;
        }
    }
}

void LowRankProjectedOptimizer::consumeDenseGradient(const ParameterId& id, const float* values,
                                                      std::size_t count) {
    auto it = dense_.find(id.name);
    if (it == dense_.end()) {
        throw std::invalid_argument("LowRankProjectedOptimizer: nessun parametro denso registrato con nome '" +
                                     id.name + "'");
    }
    DenseState& state = it->second;
    if (state.accumulator.size() != count) {
        throw std::invalid_argument("LowRankProjectedOptimizer: il parametro denso '" + id.name +
                                     "' ha una dimensione diversa dal suo gradiente");
    }
    state.touched = true;
    for (std::size_t i = 0; i < count; ++i) {
        state.accumulator[i] += values[i];
    }
}

void LowRankProjectedOptimizer::applyTernaryUpdate(TernaryState& state) {
    const std::size_t values = state.rank * state.cols;
    const float bias1 = 1.0F - std::pow(options_.beta1, static_cast<float>(step_ + 1));
    const float bias2 = 1.0F - std::pow(options_.beta2, static_cast<float>(step_ + 1));

    // Adam nel sottospazio: l'uscita ha modulo ~1 per costruzione, che
    // e' esattamente cio' che serve a un aggiornamento espresso in
    // unita' di griglia ternaria.
    std::vector<float> direction(values);
    for (std::size_t i = 0; i < values; ++i) {
        const float gradient = state.accumulator[i];
        state.firstMoment[i] = options_.beta1 * state.firstMoment[i] + (1.0F - options_.beta1) * gradient;
        state.secondMoment[i] =
            options_.beta2 * state.secondMoment[i] + (1.0F - options_.beta2) * gradient * gradient;
        const float m = state.firstMoment[i] / bias1;
        const float v = state.secondMoment[i] / bias2;
        direction[i] = m / (std::sqrt(v) + options_.eps);
    }

    if (consolidation_.enabled) {
        // Modalita' sperimentale: l'aggiornamento non tocca subito i
        // trit, si accumula nel residuo (vedi consolidate()).
        for (std::size_t i = 0; i < values; ++i) {
            state.residual[i] += -options_.learningRate * direction[i];
        }
        double norm = 0.0;
        for (float value : state.residual) {
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        stats_.residualNorm += std::sqrt(norm);
        return;
    }

    // dW_riga = P[riga, :] * direzione, ricostruito un BLOCCO DI RIGHE
    // per volta: l'aggiornamento denso completo non esiste mai, come il
    // gradiente da cui viene.
    const std::size_t blockRows = std::min<std::size_t>(state.rows, 256);
    std::vector<float> update(blockRows * state.cols);
    const ScopedMemory scope(MemoryArena::Workspace, update.size() * sizeof(float));

    for (std::size_t first = 0; first < state.rows; first += blockRows) {
        const std::size_t count = std::min(blockRows, state.rows - first);
        std::fill(update.begin(), update.begin() + static_cast<std::ptrdiff_t>(count * state.cols), 0.0F);

        for (std::size_t r = 0; r < count; ++r) {
            float* target = update.data() + r * state.cols;
            for (std::size_t j = 0; j < state.rank; ++j) {
                const float p =
                    projectionEntry(state.seed, state.projectionEpoch, first + r, j, state.rank);
                const float* source = direction.data() + j * state.cols;
                for (std::size_t c = 0; c < state.cols; ++c) {
                    target[c] += -options_.learningRate * p * source[c];
                }
            }
        }

        const TernaryUpdateStats applied =
            applyTernaryUpdateBlock(*state.weight, first, count, update.data(), state.seed,
                                     static_cast<std::uint64_t>(step_), TernaryUpdateUnits::Grid);
        stats_.ternaryFlips += applied.flips;
        stats_.ternaryElements += applied.elementsConsidered;
    }
}

void LowRankProjectedOptimizer::consolidate(TernaryState& state) {
    const std::size_t blockRows = std::min<std::size_t>(state.rows, 256);
    std::vector<float> delta(blockRows * state.cols);
    const ScopedMemory scope(MemoryArena::Workspace, delta.size() * sizeof(float));

    // Tetto sul numero di trit che una singola consolidazione puo'
    // muovere: senza, un residuo accumulato a lungo ribalterebbe mezza
    // matrice in un colpo e la loss salterebbe.
    const auto maxFlips = static_cast<std::size_t>(consolidation_.maxFlipFraction *
                                                    static_cast<float>(state.rows * state.cols));
    std::size_t flips = 0;
    const std::uint64_t consolidationSeed =
        splitMix64(state.seed ^ (static_cast<std::uint64_t>(step_) * 0xA24BAED4963EE407ULL));

    for (std::size_t first = 0; first < state.rows && flips < maxFlips; first += blockRows) {
        const std::size_t count = std::min(blockRows, state.rows - first);
        std::fill(delta.begin(), delta.begin() + static_cast<std::ptrdiff_t>(count * state.cols), 0.0F);

        for (std::size_t r = 0; r < count; ++r) {
            float* target = delta.data() + r * state.cols;
            for (std::size_t j = 0; j < state.rank; ++j) {
                const float p = projectionEntry(state.seed, state.projectionEpoch, first + r, j, state.rank);
                const float* source = state.residual.data() + j * state.cols;
                for (std::size_t c = 0; c < state.cols; ++c) {
                    target[c] += p * source[c];
                }
            }
        }

        for (std::size_t r = 0; r < count && flips < maxFlips; ++r) {
            for (std::size_t c = 0; c < state.cols; ++c) {
                const float accumulated = delta[r * state.cols + c];
                if (std::fabs(accumulated) < consolidation_.flipThreshold) {
                    continue;
                }
                const std::size_t flat = (first + r) * state.cols + c;
                const int oldTrit = state.weight->tritAt(flat);
                const float target = static_cast<float>(oldTrit) + accumulated;
                const int newTrit = consolidation_.stochasticFlip
                                         ? stochasticRoundToTrit(target, consolidationSeed, flat)
                                         : std::clamp(static_cast<int>(std::lround(target)), -1, 1);
                if (newTrit != oldTrit) {
                    state.weight->setTritAt(flat, newTrit);
                    ++flips;
                    if (flips >= maxFlips) {
                        break;
                    }
                }
            }
        }
    }

    stats_.ternaryFlips += flips;
    stats_.ternaryElements += state.rows * state.cols;
    for (float& value : state.residual) {
        value *= consolidation_.residualDecay;
    }
}

void LowRankProjectedOptimizer::applyDenseUpdate(DenseState& state) {
    const float bias1 = 1.0F - std::pow(options_.beta1, static_cast<float>(step_ + 1));
    const float bias2 = 1.0F - std::pow(options_.beta2, static_cast<float>(step_ + 1));

    // Parametri densi: Adam ordinario a stato pieno. Sono meno di un
    // millesimo del modello (0,86 M su 9,05 G), quindi proiettarli
    // farebbe risparmiare 7 MB peggiorando la stabilita' proprio dove
    // il modello e' piu' sensibile (requisito 15).
    //
    // Il passo e' RELATIVO all'ampiezza del parametro, come per i pesi
    // ternari lo e' al passo della griglia: cosi' un solo learning rate
    // ha lo stesso significato per entrambi.
    double parameterSquared = 0.0;
    for (float value : *state.values) {
        parameterSquared += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto parameterRms =
        static_cast<float>(std::sqrt(parameterSquared / static_cast<double>(state.values->size())));
    const float scale = parameterRms > 0.0F ? parameterRms : 1.0F;

    for (std::size_t i = 0; i < state.values->size(); ++i) {
        const float gradient = state.accumulator[i];
        state.firstMoment[i] = options_.beta1 * state.firstMoment[i] + (1.0F - options_.beta1) * gradient;
        state.secondMoment[i] =
            options_.beta2 * state.secondMoment[i] + (1.0F - options_.beta2) * gradient * gradient;
        const float m = state.firstMoment[i] / bias1;
        const float v = state.secondMoment[i] / bias2;
        float& parameter = (*state.values)[i];
        parameter -= options_.learningRate * scale * (m / (std::sqrt(v) + options_.eps));
        parameter -= options_.learningRate * options_.weightDecay * parameter;
    }
}

// Adam ordinario sulle scale: sono parametri continui, senza proiezione e
// senza quantizzazione. Nessun weight decay, perche' far decadere una scala
// verso zero spegnerebbe l'intero gruppo di pesi che governa.
void LowRankProjectedOptimizer::applyScaleUpdate(TernaryState& state) {
    if (state.scaleAccumulator.empty()) {
        return;
    }
    const float bias1 = 1.0F - std::pow(options_.beta1, static_cast<float>(step_ + 1));
    const float bias2 = 1.0F - std::pow(options_.beta2, static_cast<float>(step_ + 1));
    std::vector<float>& scales = state.weight->scales();
    for (std::size_t i = 0; i < state.scaleAccumulator.size(); ++i) {
        const float gradient = state.scaleAccumulator[i];
        state.scaleFirstMoment[i] =
            options_.beta1 * state.scaleFirstMoment[i] + (1.0F - options_.beta1) * gradient;
        state.scaleSecondMoment[i] =
            options_.beta2 * state.scaleSecondMoment[i] + (1.0F - options_.beta2) * gradient * gradient;
        const float m = state.scaleFirstMoment[i] / bias1;
        const float v = state.scaleSecondMoment[i] / bias2;
        scales[i] -= options_.learningRate * state.scaleRms * (m / (std::sqrt(v) + options_.eps));
        state.scaleAccumulator[i] = 0.0F;
    }
}

void LowRankProjectedOptimizer::endStep() {
    for (auto& entry : ternary_) {
        TernaryState& state = entry.second;
        if (!state.touched) {
            continue;
        }
        applyTernaryUpdate(state);
        applyScaleUpdate(state);
        std::fill(state.accumulator.begin(), state.accumulator.end(), 0.0F);
        state.touched = false;
    }

    for (auto& entry : dense_) {
        DenseState& state = entry.second;
        if (!state.touched) {
            continue;
        }
        applyDenseUpdate(state);
        std::fill(state.accumulator.begin(), state.accumulator.end(), 0.0F);
        state.touched = false;
    }

    ++step_;
    stats_.stepCount = step_;

    if (consolidation_.enabled && step_ % consolidation_.consolidationInterval == 0) {
        for (auto& entry : ternary_) {
            consolidate(entry.second);
        }
        ++stats_.consolidations;
    }

    if (step_ % options_.projectionInterval == 0) {
        // Riseminatura del sottospazio: i momenti vengono azzerati
        // perche' si riferiscono alla base VECCHIA. Conservarli
        // significherebbe sommare coordinate che non descrivono piu' le
        // stesse direzioni — un errore silenzioso che si manifesta solo
        // come "l'addestramento peggiora ogni tot passi".
        for (auto& entry : ternary_) {
            TernaryState& state = entry.second;
            ++state.projectionEpoch;
            std::fill(state.firstMoment.begin(), state.firstMoment.end(), 0.0F);
            std::fill(state.secondMoment.begin(), state.secondMoment.end(), 0.0F);
            if (!state.residual.empty()) {
                // Il residuo vive nella base vecchia: va consolidato
                // prima di cambiarla, altrimenti la plasticita'
                // accumulata verrebbe reinterpretata in direzioni
                // sbagliate.
                consolidate(state);
                std::fill(state.residual.begin(), state.residual.end(), 0.0F);
            }
        }
        ++stats_.projectionReseeds;
    }
}

std::size_t LowRankProjectedOptimizer::stateBytes() const {
    std::size_t total = 0;
    for (const auto& entry : ternary_) {
        const TernaryState& state = entry.second;
        total += (state.firstMoment.size() + state.secondMoment.size() + state.accumulator.size() +
                   state.residual.size()) *
                  sizeof(float);
    }
    for (const auto& entry : dense_) {
        const DenseState& state = entry.second;
        total += (state.firstMoment.size() + state.secondMoment.size() + state.accumulator.size()) * sizeof(float);
    }
    return total;
}

std::size_t LowRankProjectedOptimizer::conventionalStateBytes() const {
    std::size_t total = 0;
    for (const auto& entry : ternary_) {
        total += 2 * entry.second.rows * entry.second.cols * sizeof(float);
    }
    for (const auto& entry : dense_) {
        total += 2 * entry.second.firstMoment.size() * sizeof(float);
    }
    return total;
}

namespace {

template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("stato dell'ottimizzatore: stream terminato prima del previsto");
    }
    return value;
}

void writeVector(std::ostream& out, const std::vector<float>& values) {
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void readVector(std::istream& in, std::vector<float>& values) {
    const auto count = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    if (count != values.size()) {
        throw std::runtime_error("stato dell'ottimizzatore: dimensione di un buffer diversa da quella attesa (" +
                                  std::to_string(count) + " invece di " + std::to_string(values.size()) + ")");
    }
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("stato dell'ottimizzatore: stream terminato prima del previsto");
    }
}

}  // namespace

void LowRankProjectedOptimizer::serializeState(std::ostream& out) const {
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(step_));
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(ternary_.size()));
    for (const auto& entry : ternary_) {
        writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(entry.first.size()));
        out.write(entry.first.data(), static_cast<std::streamsize>(entry.first.size()));
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(entry.second.rank));
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(entry.second.projectionEpoch));
        writeVector(out, entry.second.firstMoment);
        writeVector(out, entry.second.secondMoment);
        writeVector(out, entry.second.residual);
        // Momenti Adam delle scale: senza, una ripresa da checkpoint
        // ripartirebbe con lo slancio azzerato e non riprodurrebbe il run
        // ininterrotto.
        writeVector(out, entry.second.scaleFirstMoment);
        writeVector(out, entry.second.scaleSecondMoment);
        // scaleRms e' misurato alla registrazione sulle scale di allora.
        // Ricalcolarlo dopo una ripresa lo misurerebbe su scale gia'
        // addestrate, dando un fattore diverso da quello del run
        // ininterrotto: va salvato, non ricostruito.
        writeScalar<float>(out, entry.second.scaleRms);
    }
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(dense_.size()));
    for (const auto& entry : dense_) {
        writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(entry.first.size()));
        out.write(entry.first.data(), static_cast<std::streamsize>(entry.first.size()));
        writeVector(out, entry.second.firstMoment);
        writeVector(out, entry.second.secondMoment);
    }
}

void LowRankProjectedOptimizer::deserializeState(std::istream& in) {
    step_ = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    stats_.stepCount = step_;

    const auto ternaryCount = readScalar<std::uint32_t>(in);
    for (std::uint32_t i = 0; i < ternaryCount; ++i) {
        const auto nameLength = readScalar<std::uint32_t>(in);
        std::string name(nameLength, '\0');
        in.read(name.data(), nameLength);
        auto it = ternary_.find(name);
        if (it == ternary_.end()) {
            throw std::runtime_error("stato dell'ottimizzatore: parametro ternario '" + name +
                                      "' non registrato nel modello corrente");
        }
        const auto rank = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
        if (rank != it->second.rank) {
            throw std::runtime_error("stato dell'ottimizzatore: il parametro '" + name + "' aveva rango " +
                                      std::to_string(rank) + ", ora e' " + std::to_string(it->second.rank));
        }
        it->second.projectionEpoch = readScalar<std::uint64_t>(in);
        readVector(in, it->second.firstMoment);
        readVector(in, it->second.secondMoment);
        readVector(in, it->second.residual);
        readVector(in, it->second.scaleFirstMoment);
        readVector(in, it->second.scaleSecondMoment);
        it->second.scaleRms = readScalar<float>(in);
    }

    const auto denseCount = readScalar<std::uint32_t>(in);
    for (std::uint32_t i = 0; i < denseCount; ++i) {
        const auto nameLength = readScalar<std::uint32_t>(in);
        std::string name(nameLength, '\0');
        in.read(name.data(), nameLength);
        auto it = dense_.find(name);
        if (it == dense_.end()) {
            throw std::runtime_error("stato dell'ottimizzatore: parametro denso '" + name +
                                      "' non registrato nel modello corrente");
        }
        readVector(in, it->second.firstMoment);
        readVector(in, it->second.secondMoment);
    }
}

void LowRankProjectedOptimizer::resetStats() {
    const std::size_t steps = stats_.stepCount;
    stats_ = LowRankOptimizerStats{};
    stats_.stepCount = steps;
}

}  // namespace blackforge::blackbit
