#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "blackforge/blackbit/config.hpp"
#include "blackforge/blackbit/low_rank_optimizer.hpp"
#include "blackforge/blackbit/model.hpp"

namespace blackforge::blackbit::cuda {

struct TrainingOptions {
    std::string datasetPath;
    std::string tokenizerPath;
    std::string fromCheckpoint;
    std::string saveCheckpoint;
    std::size_t steps = 100;
    std::size_t microBatch = 1;
    std::size_t maxVramMb = 7800;
    unsigned int seed = 42;
    // Scheduler del learning rate. warmupSteps a 0 disattiva il warmup;
    // scheduleSteps a 0 significa "usa steps". Su un run ripreso da
    // checkpoint scheduleSteps deve valere il totale pianificato dell'INTERO
    // addestramento, non i passi di questa invocazione, altrimenti il coseno
    // riparte da capo a ogni ripresa.
    // Salvataggio periodico: 0 = solo alla fine del run. Il file viene scritto
    // su un temporaneo e poi rinominato sopra la destinazione, cosi' una
    // interruzione durante la scrittura non distrugge il checkpoint buono.
    std::size_t checkpointEvery = 0;
    std::size_t warmupSteps = 0;
    std::size_t scheduleSteps = 0;
    float minLearningRateRatio = 0.1F;
    LowRankOptimizerOptions optimizer;
    BlackBitRuntimeOptions runtime;
};

struct TrainingResult {
    std::uint64_t initialStep = 0;
    std::uint64_t finalStep = 0;
    std::uint64_t initialTokens = 0;
    std::uint64_t finalTokens = 0;
    float initialLoss = 0.0F;
    float finalLoss = 0.0F;
    std::vector<float> lossCurve;
    double stepMilliseconds = 0.0;
    double tokensPerSecond = 0.0;
    std::size_t actualPeakBytes = 0;
    std::size_t averageActualBytes = 0;
    std::size_t ternaryFlips = 0;
    std::size_t nanInfCount = 0;
    double routingEntropy = 0.0;
    double maxExpertUtilization = 0.0;
    std::size_t droppedAssignments = 0;
    std::size_t checkpointBytes = 0;
    std::string checkpointPath;
    std::string manifestPath;

    [[nodiscard]] std::string report() const;
};

TrainingResult train(const BlackBitConfig& config, const TrainingOptions& options,
                     int device = 0, std::ostream* progress = nullptr);

// Valutazione su dati mai visti.
//
// Senza questa, dopo il confine di epoca la loss di addestramento misura
// memorizzazione e generalizzazione insieme, e le due non si separano piu'.
struct EvalOptions {
    std::string datasetPath;
    std::string checkpoint;
    std::size_t microBatch = 16;
    std::size_t maxExamples = 0;   // 0 = tutto lo shard
    // Byte di testo originale per token, misurato sul corpus di valutazione.
    // Serve a convertire la loss in bit per byte, che e' l'unica metrica
    // confrontabile fra modelli con tokenizzatori diversi: i nats per token
    // dipendono da quanto e' fine la segmentazione, i bit per byte no.
    double bytesPerToken = 0.0;
    unsigned int seed = 42;
    std::size_t maxVramMb = 7800;
    BlackBitRuntimeOptions runtime;
};

struct EvalResult {
    double loss = 0.0;
    double perplexity = 0.0;
    double bitsPerByte = 0.0;   // 0 se bytesPerToken non e' stato fornito
    std::uint64_t scoredTokens = 0;
    std::size_t examples = 0;
    std::uint64_t checkpointStep = 0;
    std::uint64_t checkpointTokens = 0;
    double seconds = 0.0;
};

EvalResult evaluate(const EvalOptions& options, int device = 0, std::ostream* progress = nullptr);

}  // namespace blackforge::blackbit::cuda
