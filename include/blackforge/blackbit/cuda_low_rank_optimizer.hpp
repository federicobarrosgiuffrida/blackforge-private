#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>

#include "blackforge/blackbit/cuda_memory.hpp"
#include "blackforge/blackbit/cuda_ternary.hpp"
#include "blackforge/blackbit/cuda_ternary_linear.hpp"
#include "blackforge/blackbit/low_rank_optimizer.hpp"

namespace blackforge::blackbit::cuda {

struct LowRankOptimizerStepStats {
    std::size_t elements = 0;
    std::size_t flips = 0;
    std::size_t positiveFlips = 0;
    std::size_t negativeFlips = 0;
    std::size_t saturatedTrits = 0;
    std::size_t nanInfCount = 0;
    double gradientRms = 0.0;
    double optimizerNorm = 0.0;
    double updateRms = 0.0;

    [[nodiscard]] double flipFraction() const {
        return elements == 0 ? 0.0 : static_cast<double>(flips) / static_cast<double>(elements);
    }
};

struct LowRankOptimizerStats {
    std::size_t stepCount = 0;
    std::size_t totalFlips = 0;
    std::size_t totalElements = 0;
    std::size_t projectionReseeds = 0;
    LowRankOptimizerStepStats lastStep;
};

class LowRankProjectedOptimizer final : public GradientSink {
public:
    explicit LowRankProjectedOptimizer(LowRankOptimizerOptions options = {});

    LowRankProjectedOptimizer(const LowRankProjectedOptimizer&) = delete;
    LowRankProjectedOptimizer& operator=(const LowRankProjectedOptimizer&) = delete;

    void registerTernary(const std::string& name, cuda::TernaryTensor& weight);
    void registerDense(const std::string& name, Tensor& values);
    void setRankFor(const std::string& name, std::size_t rank);

    void consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                     const float* deviceBlock) override;
    void consumeDenseGradient(const ParameterId& id, const float* deviceValues, std::size_t count) override;
    void endStep();

    [[nodiscard]] std::size_t stateBytes() const;
    [[nodiscard]] std::size_t conventionalStateBytes() const;
    [[nodiscard]] std::size_t stepCount() const { return step_; }
    [[nodiscard]] const LowRankOptimizerStats& stats() const { return stats_; }

    void serializeState(std::ostream& out) const;
    void deserializeState(std::istream& in);

    void setLearningRate(float value) { options_.learningRate = value; }
    [[nodiscard]] float learningRate() const { return options_.learningRate; }

private:
    struct State {
        cuda::TernaryTensor* weight = nullptr;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::size_t rank = 0;
        std::uint64_t seed = 0;
        std::uint64_t projectionEpoch = 0;
        Buffer firstMoment;
        Buffer secondMoment;
        Buffer accumulator;
        // Le scale per gruppo sono l'unica parte continua di un peso
        // ternario: senza di loro ogni peso resta inchiodato ai tre valori
        // decisi all'inizializzazione e la rete non puo' regolare
        // l'ampiezza di nessun layer. Sono una ogni groupSize pesi, quindi
        // il loro stato Adam costa 1/groupSize di uno stato denso.
        Buffer scaleFirstMoment;
        Buffer scaleSecondMoment;
        Buffer scaleAccumulator;
        // Residuo di arrotondamento, un float per peso. Allocato solo se
        // options_.errorFeedback e' attivo.
        Buffer tritResidual;
        std::size_t scaleCount = 0;
        // Magnitudine tipica delle scale, misurata all'inizializzazione:
        // rende l'aggiornamento relativo invece che assoluto.
        float scaleRms = 1.0F;
        bool touched = false;
    };

    struct DenseState {
        Tensor* values = nullptr;
        Buffer firstMoment;
        Buffer secondMoment;
        Buffer accumulator;
        bool touched = false;
    };

    State& stateFor(const ParameterId& id);
    void allocateState(const std::string& name, State& state, cuda::TernaryTensor& weight);

    LowRankOptimizerOptions options_;
    std::size_t step_ = 0;
    LowRankOptimizerStats stats_;
    std::unordered_map<std::string, std::size_t> rankOverrides_;
    std::unordered_map<std::string, State> states_;
    std::unordered_map<std::string, DenseState> denseStates_;
    Buffer deviceStats_;
};

}  // namespace blackforge::blackbit::cuda
