#include "blackforge/blackbit/cuda_model.hpp"

#include "blackforge/blackbit/cuda_profile.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void initializeHeadStatsKernel(float* rowMax, float* rowSum, float* targetLogit,
                                          std::size_t tokens) {
    const std::size_t token = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token < tokens) {
        rowMax[token] = -3.402823466e+38F;
        rowSum[token] = 0.0F;
        targetLogit[token] = 0.0F;
    }
}

__global__ void updateHeadStatsKernel(float* rowMax, float* rowSum, float* targetLogit,
                                      const float* logits, const int* targets, std::size_t tokens,
                                      std::size_t firstVocabulary, std::size_t count) {
    const std::size_t token = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= tokens || targets[token] < 0) return;
    float maximum = rowMax[token];
    float total = rowSum[token];
    for (std::size_t column = 0; column < count; ++column) {
        const float value = logits[token * count + column];
        if (value > maximum) {
            total *= expf(maximum - value);
            maximum = value;
        }
        total += expf(value - maximum);
    }
    rowMax[token] = maximum;
    rowSum[token] = total;
    const std::size_t target = static_cast<std::size_t>(targets[token]);
    if (target >= firstVocabulary && target < firstVocabulary + count) {
        targetLogit[token] = logits[token * count + target - firstVocabulary];
    }
}

struct LossStats {
    double sum;
    unsigned long long scored;
    unsigned long long nanInf;
    unsigned long long invalid;
};

__global__ void headLossKernel(const float* rowMax, const float* rowSum, const float* targetLogit,
                               const int* targets, std::size_t tokens, std::size_t vocabulary,
                               LossStats* stats) {
    const std::size_t token = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= tokens || targets[token] < 0) return;
    if (static_cast<std::size_t>(targets[token]) >= vocabulary) {
        atomicAdd(&stats->invalid, 1ULL);
        return;
    }
    const float loss = rowMax[token] + logf(rowSum[token]) - targetLogit[token];
    if (isfinite(loss)) atomicAdd(&stats->sum, static_cast<double>(loss));
    else atomicAdd(&stats->nanInf, 1ULL);
    atomicAdd(&stats->scored, 1ULL);
}

__global__ void headGradientKernel(float* gradient, const float* logits, const float* rowMax,
                                   const float* rowSum, const int* targets, std::size_t tokens,
                                   std::size_t firstVocabulary, std::size_t count,
                                   float inverseScored) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * count) return;
    const std::size_t token = index / count;
    const std::size_t column = index % count;
    if (targets[token] < 0) {
        gradient[index] = 0.0F;
        return;
    }
    float value = expf(logits[index] - rowMax[token]) / rowSum[token];
    if (firstVocabulary + column == static_cast<std::size_t>(targets[token])) value -= 1.0F;
    gradient[index] = value * inverseScored;
}

__global__ void addLookupGradientKernel(float* weightGradient, const float* lookupGradient,
                                        const int* tokenIds, std::size_t tokens, std::size_t hidden,
                                        std::size_t firstVocabulary, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) return;
    const std::size_t token = index / hidden;
    const std::size_t column = index % hidden;
    const std::size_t id = static_cast<std::size_t>(tokenIds[token]);
    if (id >= firstVocabulary && id < firstVocabulary + count) {
        atomicAdd(weightGradient + (id - firstVocabulary) * hidden + column, lookupGradient[index]);
    }
}

}  // namespace

BlackBitBlock::BlackBitBlock(blackforge::blackbit::BlackBitBlock& cpuReference,
                             const BlackBitConfig& config)
    : attentionNormName_(cpuReference.attentionNorm().name()),
      moeNormName_(cpuReference.moeNorm().name()),
      attentionGamma_(Tensor::fromHost(runtime::Tensor(
          {config.hiddenSize}, cpuReference.attentionNorm().gamma()), MemoryArena::DenseParameters)),
      moeGamma_(Tensor::fromHost(runtime::Tensor(
          {config.hiddenSize}, cpuReference.moeNorm().gamma()), MemoryArena::DenseParameters)),
      attention_(cpuReference.attention(), config),
      moe_(cpuReference.moe(), config) {}

Tensor BlackBitBlock::forward(const Tensor& input, BlockCache& cache) const {
    GpuProfiler::instance().begin(GpuPhase::Norm);
    cache.normedForAttention = rmsNormForward(input, attentionGamma_, cache.attentionNorm);
    GpuProfiler::instance().end(GpuPhase::Norm);
    GpuProfiler::instance().begin(GpuPhase::Attention);
    Tensor attended = attention_.forward(cache.normedForAttention, cache.attention);
    GpuProfiler::instance().end(GpuPhase::Attention);
    cache.afterAttention = add(input, attended, MemoryArena::Activations);
    GpuProfiler::instance().begin(GpuPhase::Norm);
    cache.normedForMoE = rmsNormForward(cache.afterAttention, moeGamma_, cache.moeNorm);
    GpuProfiler::instance().end(GpuPhase::Norm);
    Tensor mixed = moe_.forward(cache.normedForMoE, cache.moe, cache.routing);
    return add(cache.afterAttention, mixed, MemoryArena::Activations);
}

Tensor BlackBitBlock::backward(const Tensor& input, const Tensor& gradOutput, const BlockCache& cache,
                               GradientSink* sink) const {
    Tensor gradNormedMoe = moe_.backward(cache.normedForMoE, gradOutput, cache.moe, cache.routing, sink);
    GpuProfiler::instance().begin(GpuPhase::Norm);
    RmsNormGrad moeNormGrad = rmsNormBackward(cache.afterAttention, moeGamma_, gradNormedMoe, cache.moeNorm);
    GpuProfiler::instance().end(GpuPhase::Norm);
    if (sink != nullptr) {
        sink->consumeDenseGradient({moeNormName_, 1, moeGamma_.elementCount()}, moeNormGrad.gamma.data(),
                                   moeNormGrad.gamma.elementCount());
    }
    Tensor gradAfterAttention = add(gradOutput, moeNormGrad.input, MemoryArena::Activations);
    GpuProfiler::instance().begin(GpuPhase::Attention);
    Tensor gradNormedAttention = attention_.backward(cache.normedForAttention, gradAfterAttention,
                                                     cache.attention, sink);
    GpuProfiler::instance().end(GpuPhase::Attention);
    GpuProfiler::instance().begin(GpuPhase::Norm);
    RmsNormGrad attentionNormGrad = rmsNormBackward(input, attentionGamma_, gradNormedAttention,
                                                    cache.attentionNorm);
    GpuProfiler::instance().end(GpuPhase::Norm);
    if (sink != nullptr) {
        sink->consumeDenseGradient({attentionNormName_, 1, attentionGamma_.elementCount()},
                                   attentionNormGrad.gamma.data(), attentionNormGrad.gamma.elementCount());
    }
    addInPlace(gradAfterAttention, attentionNormGrad.input);
    return gradAfterAttention;
}

BlackBitModel::BlackBitModel(blackforge::blackbit::BlackBitModel& cpuReference)
    : config_(cpuReference.config()),
      embedding_(cpuReference.embedding()),
      finalNormName_(cpuReference.finalNorm().name()),
      finalGamma_(Tensor::fromHost(runtime::Tensor(
          {config_.hiddenSize}, cpuReference.finalNorm().gamma()), MemoryArena::DenseParameters)),
      options_(cpuReference.runtimeOptions()) {
    blocks_.reserve(config_.numLayers);
    for (auto& block : cpuReference.blocks()) blocks_.emplace_back(block, config_);
}

Tensor BlackBitModel::embedTokens(const IndexTensor& tokenIds) const {
    const GpuPhaseScope scope(GpuPhase::Embedding);
    return embeddingLookup(tokenIds, embedding_.weight(), config_.hiddenSize);
}

Tensor BlackBitModel::forwardHidden(const std::vector<int>& tokenIds, std::size_t batch, std::size_t seq,
                                    ForwardCache& cache) const {
    if (tokenIds.size() != batch * seq) throw std::invalid_argument("CUDA BlackBitModel: token shape mismatch");
    const IndexTensor ids = IndexTensor::fromHost({batch, seq}, tokenIds);
    Tensor hidden = embedTokens(ids);
    cache.blockInputs.clear();
    cache.blocks.clear();
    cache.blockInputs.reserve(blocks_.size());
    cache.blocks.resize(blocks_.size());
    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        cache.blockInputs.push_back(hidden.clone(MemoryArena::Activations));
        hidden = blocks_[layer].forward(hidden, cache.blocks[layer]);
    }
    cache.preNormHidden = hidden.clone(MemoryArena::Activations);
    return rmsNormForward(hidden, finalGamma_, cache.finalNorm);
}

void BlackBitModel::setVocabChunk(std::size_t chunk) {
    if (chunk == 0) throw std::invalid_argument("CUDA BlackBitModel: vocab chunk must be positive");
    options_.vocabChunk = std::min(chunk, config_.vocabSize);
}

void BlackBitModel::setRuntimeOptions(const BlackBitRuntimeOptions& options) {
    if (options.recomputeEveryN == 0) throw std::invalid_argument("CUDA BlackBitModel: recomputeEveryN must be positive");
    options_ = options;
    setVocabChunk(options.vocabChunk);
}

BlackBitStepResult BlackBitModel::trainStep(const std::vector<int>& tokenIds,
                                            const std::vector<int>& targets, std::size_t batch,
                                            std::size_t seq, GradientSink* sink) {
    const std::size_t tokens = batch * seq;
    if (tokenIds.size() != tokens || targets.size() != tokens || seq > config_.maxSeqLen) {
        throw std::invalid_argument("CUDA BlackBitModel: invalid training shape");
    }
    const IndexTensor ids = IndexTensor::fromHost({batch, seq}, tokenIds);
    const IndexTensor deviceTargets = IndexTensor::fromHost({tokens}, targets);
    BlackBitStepResult result;
    std::vector<Tensor> checkpoints;
    checkpoints.reserve(blocks_.size());
    std::vector<BlockCache> savedCaches;
    if (options_.recompute == ActivationRecompute::None) savedCaches.resize(blocks_.size());
    Tensor activations = embedTokens(ids);
    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        checkpoints.push_back(activations.clone(MemoryArena::Activations));
        if (options_.recompute == ActivationRecompute::None) {
            activations = blocks_[layer].forward(activations, savedCaches[layer]);
            result.routing.push_back(savedCaches[layer].routing);
        } else {
            BlockCache scratch;
            activations = blocks_[layer].forward(activations, scratch);
            result.routing.push_back(scratch.routing);
        }
    }
    Tensor preNormHidden = activations.clone(MemoryArena::Activations);
    RmsNormCache finalNormCache;
    GpuProfiler::instance().begin(GpuPhase::Norm);
    Tensor finalHidden = rmsNormForward(activations, finalGamma_, finalNormCache);
    GpuProfiler::instance().end(GpuPhase::Norm);
    for (const MoERoutingStats& routing : result.routing) result.auxiliaryLoss += routing.loadBalancingLoss;
    if (!result.routing.empty()) result.auxiliaryLoss /= static_cast<float>(result.routing.size());

    const std::size_t chunk = std::min(options_.vocabChunk, config_.vocabSize);
    Tensor rowMax({tokens}, MemoryArena::Temporary);
    Tensor rowSum({tokens}, MemoryArena::Temporary);
    Tensor targetLogit({tokens}, MemoryArena::Temporary);
    // Le tre passate sulla testa di uscita sono la parte piu' costosa in
    // FLOP del passo (vocab 65536 x hidden 3072, ripetuta tre volte), ma
    // non e' evidente quanto tempo GPU consumino davvero: si misura.
    GpuProfiler::instance().begin(GpuPhase::Head);
    initializeHeadStatsKernel<<<gridFor(tokens), kBlockSize>>>(
        rowMax.data(), rowSum.data(), targetLogit.data(), tokens);
    for (std::size_t first = 0; first < config_.vocabSize; first += chunk) {
        const std::size_t count = std::min(chunk, config_.vocabSize - first);
        Tensor logits = embedding_.forwardRows(finalHidden, first, count);
        updateHeadStatsKernel<<<gridFor(tokens), kBlockSize>>>(
            rowMax.data(), rowSum.data(), targetLogit.data(), logits.data(), deviceTargets.data(), tokens,
            first, count);
    }
    Buffer lossStats(sizeof(LossStats), MemoryArena::Temporary);
    BLACKFORGE_CUDA_CHECK(cudaMemset(lossStats.data(), 0, lossStats.bytes()));
    headLossKernel<<<gridFor(tokens), kBlockSize>>>(
        rowMax.data(), rowSum.data(), targetLogit.data(), deviceTargets.data(), tokens,
        config_.vocabSize, lossStats.as<LossStats>());
    GpuProfiler::instance().end(GpuPhase::Head);
    LossStats hostLoss{};
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(&hostLoss, lossStats.data(), sizeof(hostLoss), cudaMemcpyDeviceToHost));
    if (hostLoss.invalid != 0) throw std::invalid_argument("CUDA BlackBitModel: target outside vocabulary");
    result.scoredTokens = static_cast<std::size_t>(hostLoss.scored);
    result.loss = hostLoss.scored == 0 ? 0.0F : static_cast<float>(hostLoss.sum / hostLoss.scored);
    if (hostLoss.nanInf != 0) {
        result.sawNaN = true;
        result.firstUnstableTensor = "cross_entropy";
        return result;
    }
    if (sink == nullptr || hostLoss.scored == 0) return result;

    const float inverseScored = 1.0F / static_cast<float>(hostLoss.scored);
    Tensor gradHidden = Tensor::zeros(finalHidden.shape(), MemoryArena::Activations);
    GpuProfiler::instance().begin(GpuPhase::Head);
    for (std::size_t first = 0; first < config_.vocabSize; first += chunk) {
        const std::size_t count = std::min(chunk, config_.vocabSize - first);
        Tensor logits = embedding_.forwardRows(finalHidden, first, count);
        Tensor gradLogits(logits.shape(), MemoryArena::GradientTiles);
        headGradientKernel<<<gridFor(gradLogits.elementCount()), kBlockSize>>>(
            gradLogits.data(), logits.data(), rowMax.data(), rowSum.data(), deviceTargets.data(), tokens,
            first, count, inverseScored);
        Tensor contribution = embedding_.backwardInputRows(gradLogits, first);
        addInPlace(gradHidden, contribution);
    }
    GpuProfiler::instance().end(GpuPhase::Head);
    GpuProfiler::instance().begin(GpuPhase::Norm);
    RmsNormGrad finalNormGrad = rmsNormBackward(preNormHidden, finalGamma_, gradHidden, finalNormCache);
    GpuProfiler::instance().end(GpuPhase::Norm);
    sink->consumeDenseGradient({finalNormName_, 1, config_.hiddenSize}, finalNormGrad.gamma.data(),
                               finalNormGrad.gamma.elementCount());
    Tensor gradBlockInput = std::move(finalNormGrad.input);
    for (std::size_t layer = blocks_.size(); layer-- > 0;) {
        if (options_.recompute == ActivationRecompute::None) {
            gradBlockInput = blocks_[layer].backward(checkpoints[layer], gradBlockInput,
                                                     savedCaches[layer], sink);
        } else {
            BlockCache recomputed;
            (void)blocks_[layer].forward(checkpoints[layer], recomputed);
            gradBlockInput = blocks_[layer].backward(checkpoints[layer], gradBlockInput, recomputed, sink);
        }
    }
    GradientLifetimeStats& lifetime = gradientLifetimeStats();
    GpuProfiler::instance().begin(GpuPhase::Head);
    for (std::size_t first = 0; first < config_.vocabSize; first += chunk) {
        const std::size_t count = std::min(chunk, config_.vocabSize - first);
        Tensor logits = embedding_.forwardRows(finalHidden, first, count);
        Tensor gradLogits(logits.shape(), MemoryArena::GradientTiles);
        headGradientKernel<<<gridFor(gradLogits.elementCount()), kBlockSize>>>(
            gradLogits.data(), logits.data(), rowMax.data(), rowSum.data(), deviceTargets.data(), tokens,
            first, count, inverseScored);
        Tensor weightGradient = embedding_.weightGradientRows(finalHidden, gradLogits, first);
        addLookupGradientKernel<<<gridFor(tokens * config_.hiddenSize), kBlockSize>>>(
            weightGradient.data(), gradBlockInput.data(), ids.data(), tokens, config_.hiddenSize, first, count);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        lifetime.liveBytes = weightGradient.bytes();
        lifetime.peakLiveBytes = std::max(lifetime.peakLiveBytes, lifetime.liveBytes);
        lifetime.cumulativeBytes += weightGradient.bytes();
        ++lifetime.blocksProduced;
        sink->consumeWeightGradientBlock(embedding_.parameterId(), first, count, weightGradient.data());
        lifetime.liveBytes = 0;
        ++lifetime.blocksReleased;
    }
    GpuProfiler::instance().end(GpuPhase::Head);
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

}  // namespace blackforge::blackbit::cuda
