#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"
#include "blackforge/blackbit/cuda_tensor.hpp"
#include "blackforge/blackbit/cuda_ternary.hpp"
#include "blackforge/blackbit/cuda_ternary_linear.hpp"
#include "blackforge/blackbit/low_rank_optimizer.hpp"
#include "blackforge/blackbit/ternary.hpp"
#include "blackforge/blackbit/ternary_linear.hpp"
#include "blackforge/runtime/tensor.hpp"

namespace bb = blackforge::blackbit;
namespace bbcuda = blackforge::blackbit::cuda;
using blackforge::runtime::Tensor;

namespace {

bb::TernaryTensor makeWeight(std::size_t rows, std::size_t cols) {
    std::vector<std::int8_t> trits(rows * cols);
    for (std::size_t i = 0; i < trits.size(); ++i) {
        trits[i] = static_cast<std::int8_t>(static_cast<int>((i * 7 + 1) % 3) - 1);
    }
    std::vector<float> scales(rows * ((cols + 19) / 20));
    for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = 0.05F + 0.002F * static_cast<float>(i);
    return bb::TernaryTensor::fromTrits({rows, cols}, trits, scales, 20);
}

std::vector<float> makeGradient(std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(static_cast<int>((i * 11) % 17) - 8) * 0.013F;
    }
    return values;
}

}  // namespace

TEST(CudaBlackBitOptimizerTest, MatchesCpuProjectedUpdateForDeterministicStep) {
    constexpr std::size_t rows = 17;
    constexpr std::size_t cols = 23;
    bb::TernaryTensor cpuWeight = makeWeight(rows, cols);
    bbcuda::TernaryTensor gpuWeight(cpuWeight);
    const std::vector<float> gradient = makeGradient(rows * cols);

    bb::LowRankOptimizerOptions options;
    options.learningRate = 0.2F;
    options.rank = 5;
    options.projectionInterval = 100;
    options.seed = 0x12345678ULL;

    bb::LowRankProjectedOptimizer cpu(options);
    cpu.registerTernary("weight", cpuWeight);
    cpu.consumeWeightGradientBlock({"weight", rows, cols}, 0, rows, gradient.data());
    cpu.endStep();

    bbcuda::LowRankProjectedOptimizer gpu(options);
    gpu.registerTernary("weight", gpuWeight);
    const Tensor hostGradient({rows, cols}, gradient);
    const auto deviceGradient = bbcuda::Tensor::fromHost(hostGradient, bbcuda::MemoryArena::GradientTiles);
    gpu.consumeWeightGradientBlock({"weight", rows, cols}, 0, rows, deviceGradient.data());
    gpu.endStep();
    const bb::TernaryTensor downloaded = gpuWeight.download();

    EXPECT_EQ(downloaded.packedWords(), cpuWeight.packedWords());
    // Le scale sono ora addestrate, quindi non sono piu' costanti: sono
    // valori continui prodotti da due riduzioni con ordini di somma diversi
    // (riduzione warp su GPU, ciclo sequenziale su CPU). L'uguaglianza
    // esatta valeva solo finche' restavano ferme al valore iniziale; qui la
    // proprieta' vera e' che i due percorsi calcolino lo stesso
    // aggiornamento, a meno dell'arrotondamento. I trit, che sono interi,
    // restano confrontati per uguaglianza esatta sopra.
    ASSERT_EQ(downloaded.scales().size(), cpuWeight.scales().size());
    for (std::size_t i = 0; i < downloaded.scales().size(); ++i) {
        EXPECT_NEAR(downloaded.scales()[i], cpuWeight.scales()[i],
                    1e-6F * std::max(1.0F, std::abs(cpuWeight.scales()[i])))
            << "scala " << i;
    }
    EXPECT_EQ(gpu.stats().lastStep.flips, cpu.stats().ternaryFlips);
    EXPECT_EQ(gpu.stats().lastStep.positiveFlips + gpu.stats().lastStep.negativeFlips,
              gpu.stats().lastStep.flips);
    EXPECT_EQ(gpu.stats().lastStep.nanInfCount, 0U);
    EXPECT_GT(gpu.stats().lastStep.gradientRms, 0.0);
    EXPECT_GT(gpu.stats().lastStep.optimizerNorm, 0.0);
    EXPECT_GT(gpu.stats().lastStep.updateRms, 0.0);
}

TEST(CudaBlackBitOptimizerTest, StateIsLowRankAndNoDenseGradientPersists) {
    constexpr std::size_t rows = 4096;
    constexpr std::size_t cols = 64;
    bb::TernaryTensor host = makeWeight(rows, cols);
    bbcuda::TernaryTensor weight(host);
    bb::LowRankOptimizerOptions options;
    options.rank = 8;
    bbcuda::LowRankProjectedOptimizer optimizer(options);
    optimizer.registerTernary("large", weight);

    EXPECT_LT(optimizer.stateBytes(), optimizer.conventionalStateBytes() / 100);
    EXPECT_GE(optimizer.stateBytes(), 3 * options.rank * cols * sizeof(float));
    EXPECT_EQ(bbcuda::gradientLifetimeStats().liveBytes, 0U);
}

TEST(CudaBlackBitOptimizerTest, DenseParametersMatchCpuAdamUpdate) {
    bb::LowRankOptimizerOptions options;
    options.learningRate = 0.07F;
    options.weightDecay = 0.02F;
    std::vector<float> cpuValues{0.7F, -0.3F, 1.1F, 0.2F, -0.8F, 0.45F};
    const std::vector<float> gradient{0.2F, -0.1F, 0.04F, 0.3F, -0.2F, 0.11F};
    bb::LowRankProjectedOptimizer cpu(options);
    cpu.registerDense("router", cpuValues);
    cpu.consumeDenseGradient({"router", 2, 3}, gradient.data(), gradient.size());
    cpu.endStep();

    const Tensor initial({2, 3}, {0.7F, -0.3F, 1.1F, 0.2F, -0.8F, 0.45F});
    auto gpuValues = bbcuda::Tensor::fromHost(initial, bbcuda::MemoryArena::DenseParameters);
    const Tensor hostGradient({2, 3}, gradient);
    const auto gpuGradient = bbcuda::Tensor::fromHost(hostGradient, bbcuda::MemoryArena::GradientTiles);
    bbcuda::LowRankProjectedOptimizer gpu(options);
    const std::size_t fixedOptimizerBytes = gpu.stateBytes();
    gpu.registerDense("router", gpuValues);
    gpu.consumeDenseGradient({"router", 2, 3}, gpuGradient.data(), gradient.size());
    gpu.endStep();

    const Tensor actual = gpuValues.toHost();
    for (std::size_t i = 0; i < cpuValues.size(); ++i) {
        EXPECT_NEAR(actual.at(i), cpuValues[i], 2.0e-6F) << "index " << i;
    }
    EXPECT_EQ(gpu.stateBytes(), fixedOptimizerBytes + 3 * gradient.size() * sizeof(float));
    EXPECT_EQ(gpu.stats().lastStep.nanInfCount, 0U);
    EXPECT_GT(gpu.stats().lastStep.optimizerNorm, 0.0);
}

TEST(CudaBlackBitOptimizerTest, ConsumesStreamingLinearGradientAndFlipsActualPackedWeights) {
    bb::TernaryLinear cpuLinear("train.weight", 20, 29, 20, 7);
    std::vector<float> denseValues(29 * 20);
    for (std::size_t i = 0; i < denseValues.size(); ++i) {
        denseValues[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.01F;
    }
    cpuLinear.loadDense(Tensor({29, 20}, std::move(denseValues)));
    bbcuda::TernaryLinear linear(cpuLinear);
    const auto before = linear.weight().download().packedWords();

    bb::LowRankOptimizerOptions options;
    options.learningRate = 0.35F;
    options.rank = 7;
    options.projectionInterval = 2;
    bbcuda::LowRankProjectedOptimizer optimizer(options);
    optimizer.registerTernary(linear.name(), linear.weight());

    std::vector<float> xValues(4 * 20);
    std::vector<float> dyValues(4 * 29);
    for (std::size_t i = 0; i < xValues.size(); ++i) xValues[i] = 0.1F + 0.01F * static_cast<float>(i % 7);
    for (std::size_t i = 0; i < dyValues.size(); ++i) dyValues[i] = 0.03F * static_cast<float>((i % 5) + 1);
    const auto input = bbcuda::Tensor::fromHost(Tensor({4, 20}, xValues));
    const auto gradOutput = bbcuda::Tensor::fromHost(Tensor({4, 29}, dyValues));

    bbcuda::resetGradientLifetimeStats();
    auto gradInput = linear.backward(input, gradOutput, &optimizer);
    (void)gradInput;
    optimizer.endStep();
    const auto after = linear.weight().download().packedWords();

    EXPECT_NE(after, before);
    EXPECT_GT(optimizer.stats().lastStep.flips, 0U);
    EXPECT_EQ(optimizer.stats().lastStep.elements, 29 * 20);
    EXPECT_EQ(optimizer.stats().lastStep.nanInfCount, 0U);
    EXPECT_EQ(bbcuda::gradientLifetimeStats().liveBytes, 0U);
    EXPECT_LT(bbcuda::gradientLifetimeStats().peakLiveBytes, 29 * 20 * sizeof(float));

    // Second step triggers the configured projection reseed.
    auto secondGradInput = linear.backward(input, gradOutput, &optimizer);
    (void)secondGradInput;
    optimizer.endStep();
    EXPECT_EQ(optimizer.stats().projectionReseeds, 1U);
}

TEST(CudaBlackBitOptimizerTest, RejectsUnknownParametersAndBadConfiguration) {
    bb::LowRankOptimizerOptions bad;
    bad.rank = 0;
    EXPECT_THROW(bbcuda::LowRankProjectedOptimizer ignored(bad), std::invalid_argument);

    bbcuda::LowRankProjectedOptimizer optimizer;
    const auto gradient = bbcuda::Tensor::fromHost(Tensor({2, 2}, std::vector<float>(4, 1.0F)),
                                                   bbcuda::MemoryArena::GradientTiles);
    EXPECT_THROW(optimizer.consumeWeightGradientBlock({"missing", 2, 2}, 0, 2, gradient.data()),
                 std::invalid_argument);
}
