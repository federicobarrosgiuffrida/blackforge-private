#include "blackforge/blackbit/cuda_moe.hpp"

#include "blackforge/blackbit/cuda_profile.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;
constexpr std::size_t kMaximumExperts = 32;

struct RoutingSummary {
    int counts[kMaximumExperts];
    float probabilityMass[kMaximumExperts];
    double entropySum;
    unsigned long long dropped;
};

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void softmaxSmallKernel(float* probabilities, const float* logits, std::size_t tokens,
                                   std::size_t experts) {
    const std::size_t token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0) return;
    float maximum = logits[token * experts];
    for (std::size_t expert = 1; expert < experts; ++expert) {
        maximum = fmaxf(maximum, logits[token * experts + expert]);
    }
    double total = 0.0;
    for (std::size_t expert = 0; expert < experts; ++expert) {
        const float value = expf(logits[token * experts + expert] - maximum);
        probabilities[token * experts + expert] = value;
        total += value;
    }
    for (std::size_t expert = 0; expert < experts; ++expert) {
        probabilities[token * experts + expert] =
            static_cast<float>(static_cast<double>(probabilities[token * experts + expert]) / total);
    }
}

// Massa di probabilita' ed entropia non dipendono dalle assegnazioni:
// si calcolano dalle sole probabilita', quindi non c'e' motivo di
// tenerle nel kernel sequenziale. Erano tokens * experts logaritmi in
// doppia precisione su UN thread — 4096 per layer a seq 512, e il
// profilo dava routeForwardKernel a 229 ms/step per questo.
//
// Il gradiente della loss di bilanciamento si calcola dai CONTEGGI (vedi
// routingGradientKernel), non da questa massa: qui cambia solo l'ordine
// di accumulazione di due valori riportati.
__global__ void routingStatsKernel(const float* probabilities, RoutingSummary* summary, std::size_t tokens,
                                   std::size_t experts) {
    __shared__ float sharedMass[kMaximumExperts];
    __shared__ double sharedEntropy[kBlockSize / 32];
    if (threadIdx.x < experts) sharedMass[threadIdx.x] = 0.0F;
    __syncthreads();

    double localEntropy = 0.0;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t token = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; token < tokens;
         token += stride) {
        const float* row = probabilities + token * experts;
        for (std::size_t expert = 0; expert < experts; ++expert) {
            const float value = row[expert];
            atomicAdd(&sharedMass[expert], value);
            if (value > 0.0F) localEntropy -= static_cast<double>(value) * log(static_cast<double>(value));
        }
    }

    constexpr unsigned int kFullMask = 0xFFFFFFFFU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        localEntropy += __shfl_down_sync(kFullMask, localEntropy, offset);
    }
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane == 0) sharedEntropy[warp] = localEntropy;
    __syncthreads();

    if (threadIdx.x < experts && sharedMass[threadIdx.x] != 0.0F) {
        atomicAdd(&summary->probabilityMass[threadIdx.x], sharedMass[threadIdx.x]);
    }
    if (threadIdx.x == 0) {
        double blockEntropy = 0.0;
        for (int w = 0; w < static_cast<int>(kBlockSize / 32); ++w) blockEntropy += sharedEntropy[w];
        if (blockEntropy != 0.0) atomicAdd(&summary->entropySum, blockEntropy);
    }
}

// Intentionally one deterministic GPU thread. La selezione greedy riempie
// la capacita' in ordine di token, quindi e' sequenziale per costruzione:
// parallelizzarla cambierebbe QUALI esperti vengono scelti. Qui resta solo
// quella, senza piu' logaritmi ne' accumulazioni diagnostiche.
__global__ void routeForwardKernel(const float* probabilities, int* expertOfSlot, float* weightOfSlot,
                                   int* tokensOfExpert, int* slotsOfExpert, int* counts,
                                   RoutingSummary* summary, std::size_t tokens,
                                   std::size_t experts, std::size_t topK, std::size_t capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned long long droppedCount = 0;
    for (std::size_t token = 0; token < tokens; ++token) {
        const float* row = probabilities + token * experts;
        std::uint32_t selected = 0;
        float selectedMass = 0.0F;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            std::size_t best = experts;
            for (std::size_t expert = 0; expert < experts; ++expert) {
                if ((selected & (1U << expert)) != 0 ||
                    static_cast<std::size_t>(counts[expert]) >= capacity) {
                    continue;
                }
                if (best == experts || row[expert] > row[best]) best = expert;
            }
            if (best == experts) {
                expertOfSlot[token * topK + slot] = -1;
                ++droppedCount;
                continue;
            }
            selected |= 1U << best;
            expertOfSlot[token * topK + slot] = static_cast<int>(best);
            selectedMass += row[best];
        }
        if (selectedMass <= 0.0F) selectedMass = 1.0F;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            const std::size_t slotIndex = token * topK + slot;
            const int expert = expertOfSlot[slotIndex];
            if (expert < 0) continue;
            const int count = counts[expert];
            weightOfSlot[slotIndex] = row[expert] / selectedMass;
            tokensOfExpert[static_cast<std::size_t>(expert) * capacity + count] = static_cast<int>(token);
            slotsOfExpert[static_cast<std::size_t>(expert) * capacity + count] = static_cast<int>(slotIndex);
            counts[expert] = count + 1;
        }
    }
    for (std::size_t expert = 0; expert < experts; ++expert) summary->counts[expert] = counts[expert];
    summary->dropped = droppedCount;
}

__global__ void gatherRowsKernel(float* gathered, const float* input, const int* assigned,
                                 std::size_t count, std::size_t hidden) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * hidden) return;
    const std::size_t row = index / hidden;
    const std::size_t column = index % hidden;
    gathered[index] = input[static_cast<std::size_t>(assigned[row]) * hidden + column];
}

__global__ void combineRowsKernel(float* output, float* savedBySlot, const float* expertOutput,
                                  const int* assigned, const int* slots, const float* weights,
                                  std::size_t count, std::size_t hidden) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * hidden) return;
    const std::size_t row = index / hidden;
    const std::size_t column = index % hidden;
    const std::size_t token = static_cast<std::size_t>(assigned[row]);
    const std::size_t slot = static_cast<std::size_t>(slots[row]);
    const float value = expertOutput[index];
    savedBySlot[slot * hidden + column] = value;
    output[token * hidden + column] += weights[slot] * value;
}

__global__ void routingGradientKernel(float* gradProbabilities, const float* probabilities,
                                      const int* expertOfSlot, const float* weightOfSlot,
                                      const float* expertOutputOfSlot, const float* gradOutput,
                                      const int* counts, std::size_t tokens, std::size_t hidden,
                                      std::size_t experts, std::size_t topK, std::size_t routed,
                                      float auxiliaryWeight) {
    const std::size_t token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0) return;
    float selectedMass = 0.0F;
    double correction = 0.0;
    float slotGradient[8];
    for (std::size_t slot = 0; slot < topK; ++slot) {
        const std::size_t slotIndex = token * topK + slot;
        const int expert = expertOfSlot[slotIndex];
        slotGradient[slot] = 0.0F;
        if (expert < 0) continue;
        double dot = 0.0;
        for (std::size_t column = 0; column < hidden; ++column) {
            dot += static_cast<double>(gradOutput[token * hidden + column]) *
                   expertOutputOfSlot[slotIndex * hidden + column];
        }
        slotGradient[slot] = static_cast<float>(dot);
        selectedMass += probabilities[token * experts + static_cast<std::size_t>(expert)];
        correction += dot * weightOfSlot[slotIndex];
    }
    if (selectedMass > 0.0F) {
        for (std::size_t slot = 0; slot < topK; ++slot) {
            const std::size_t slotIndex = token * topK + slot;
            const int expert = expertOfSlot[slotIndex];
            if (expert >= 0) {
                gradProbabilities[token * experts + static_cast<std::size_t>(expert)] +=
                    static_cast<float>((slotGradient[slot] - correction) / selectedMass);
            }
        }
    }
    if (auxiliaryWeight > 0.0F && routed > 0) {
        for (std::size_t expert = 0; expert < experts; ++expert) {
            const double fraction = static_cast<double>(counts[expert]) / static_cast<double>(routed);
            gradProbabilities[token * experts + expert] += static_cast<float>(
                static_cast<double>(auxiliaryWeight) * experts * fraction / static_cast<double>(tokens));
        }
    }
}

__global__ void gatherWeightedGradientKernel(float* gatheredInput, float* gatheredGradient,
                                             const float* input, const float* gradOutput,
                                             const int* assigned, const int* slots,
                                             const float* weights, std::size_t count,
                                             std::size_t hidden) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * hidden) return;
    const std::size_t row = index / hidden;
    const std::size_t column = index % hidden;
    const std::size_t token = static_cast<std::size_t>(assigned[row]);
    const std::size_t slot = static_cast<std::size_t>(slots[row]);
    gatheredInput[index] = input[token * hidden + column];
    gatheredGradient[index] = weights[slot] * gradOutput[token * hidden + column];
}

__global__ void scatterAddKernel(float* output, const float* gathered, const int* assigned,
                                 std::size_t count, std::size_t hidden) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * hidden) return;
    const std::size_t row = index / hidden;
    const std::size_t column = index % hidden;
    output[static_cast<std::size_t>(assigned[row]) * hidden + column] += gathered[index];
}

__global__ void softmaxBackwardSmallKernel(float* gradLogits, const float* probabilities,
                                           const float* gradProbabilities, std::size_t tokens,
                                           std::size_t experts) {
    const std::size_t token = blockIdx.x;
    if (token >= tokens || threadIdx.x != 0) return;
    double weighted = 0.0;
    for (std::size_t expert = 0; expert < experts; ++expert) {
        weighted += static_cast<double>(gradProbabilities[token * experts + expert]) *
                    probabilities[token * experts + expert];
    }
    for (std::size_t expert = 0; expert < experts; ++expert) {
        const std::size_t index = token * experts + expert;
        gradLogits[index] = probabilities[index] *
            static_cast<float>(static_cast<double>(gradProbabilities[index]) - weighted);
    }
}

}  // namespace

MoEExpert::MoEExpert(blackforge::blackbit::MoEExpert& cpuReference)
    : gate_(cpuReference.gate()), up_(cpuReference.up()), down_(cpuReference.down()) {}

Tensor MoEExpert::forward(const Tensor& input) const {
    Tensor gate = gate_.forward(input);
    Tensor up = up_.forward(input);
    Tensor hidden = siluMultiply(gate, up);
    return down_.forward(hidden);
}

Tensor MoEExpert::backward(const Tensor& input, const Tensor& gradOutput, GradientSink* sink) const {
    Tensor gate = gate_.forward(input);
    Tensor up = up_.forward(input);
    Tensor hidden = siluMultiply(gate, up);
    Tensor gradHidden = down_.backward(hidden, gradOutput, sink);
    SiluMultiplyGrad activationGrad = siluMultiplyBackward(gate, up, gradHidden);
    Tensor fromGate = gate_.backward(input, activationGrad.gate, sink);
    Tensor fromUp = up_.backward(input, activationGrad.up, sink);
    return add(fromGate, fromUp, MemoryArena::Activations);
}

MoELayer::MoELayer(blackforge::blackbit::MoELayer& cpuReference, const BlackBitConfig& config)
    : config_(config),
      routerName_(cpuReference.router().name()),
      routerWeight_(Tensor::fromHost(runtime::Tensor(
          {config.numExperts, config.hiddenSize}, cpuReference.router().weight()), MemoryArena::DenseParameters)) {
    config_.validate();
    if (config_.numExperts > 32 || config_.expertsPerToken > 8) {
        throw std::invalid_argument("CUDA MoELayer: supports at most 32 experts and top-8 routing");
    }
    experts_.reserve(config_.numExperts);
    for (auto& expert : cpuReference.experts()) experts_.emplace_back(expert);
}

std::size_t MoELayer::capacityFor(std::size_t tokens) const {
    const double fair = static_cast<double>(tokens * config_.expertsPerToken) /
                        static_cast<double>(config_.numExperts);
    return std::max<std::size_t>(
        static_cast<std::size_t>(std::ceil(fair * static_cast<double>(config_.expertCapacityFactor))), 1);
}

Tensor MoELayer::forward(const Tensor& input, MoECache& cache, MoERoutingStats& stats) const {
    if (input.rank() < 2 || input.shape().back() != config_.hiddenSize) {
        throw std::invalid_argument("CUDA MoELayer: hidden dimension mismatch");
    }
    const std::size_t tokens = input.elementCount() / config_.hiddenSize;
    const std::size_t slots = tokens * config_.expertsPerToken;
    cache.tokens = tokens;
    cache.capacity = capacityFor(tokens);
    GpuProfiler::instance().begin(GpuPhase::Router);
    Tensor logits = denseLinear(input, routerWeight_);
    cache.probabilities = Tensor({tokens, config_.numExperts}, MemoryArena::MoeDispatch);
    softmaxSmallKernel<<<static_cast<unsigned int>(tokens), 1>>>(
        cache.probabilities.data(), logits.data(), tokens, config_.numExperts);
    cache.expertOfSlot = IndexTensor({slots}, MemoryArena::MoeDispatch);
    cache.weightOfSlot = Tensor::zeros({slots}, MemoryArena::MoeDispatch);
    cache.expertOutputOfSlot = Tensor::zeros({slots, config_.hiddenSize}, MemoryArena::MoeDispatch);
    cache.tokensOfExpert = IndexTensor({config_.numExperts, cache.capacity}, MemoryArena::MoeDispatch);
    cache.slotsOfExpert = IndexTensor({config_.numExperts, cache.capacity}, MemoryArena::MoeDispatch);
    cache.assignmentsPerExpert = IndexTensor({config_.numExperts}, MemoryArena::MoeDispatch);
    BLACKFORGE_CUDA_CHECK(cudaMemset(cache.expertOfSlot.data(), 0xFF, cache.expertOfSlot.bytes()));
    BLACKFORGE_CUDA_CHECK(cudaMemset(cache.tokensOfExpert.data(), 0xFF, cache.tokensOfExpert.bytes()));
    BLACKFORGE_CUDA_CHECK(cudaMemset(cache.slotsOfExpert.data(), 0xFF, cache.slotsOfExpert.bytes()));
    BLACKFORGE_CUDA_CHECK(cudaMemset(cache.assignmentsPerExpert.data(), 0, cache.assignmentsPerExpert.bytes()));
    Buffer summaryBuffer(sizeof(RoutingSummary), MemoryArena::Temporary);
    BLACKFORGE_CUDA_CHECK(cudaMemset(summaryBuffer.data(), 0, summaryBuffer.bytes()));
    routingStatsKernel<<<gridFor(tokens), kBlockSize>>>(
        cache.probabilities.data(), summaryBuffer.as<RoutingSummary>(), tokens, config_.numExperts);
    routeForwardKernel<<<1, 1>>>(
        cache.probabilities.data(), cache.expertOfSlot.data(), cache.weightOfSlot.data(),
        cache.tokensOfExpert.data(), cache.slotsOfExpert.data(), cache.assignmentsPerExpert.data(),
        summaryBuffer.as<RoutingSummary>(), tokens, config_.numExperts, config_.expertsPerToken,
        cache.capacity);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    GpuProfiler::instance().end(GpuPhase::Router);
    RoutingSummary hostSummary{};
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(&hostSummary, summaryBuffer.data(), sizeof(hostSummary),
                                     cudaMemcpyDeviceToHost));
    cache.hostAssignmentsPerExpert.assign(hostSummary.counts,
                                          hostSummary.counts + config_.numExperts);
    stats = MoERoutingStats{};
    stats.tokens = tokens;
    stats.assignments = slots;
    stats.droppedAssignments = static_cast<std::size_t>(hostSummary.dropped);
    stats.tokensPerExpert.resize(config_.numExperts);
    for (std::size_t expert = 0; expert < config_.numExperts; ++expert) {
        stats.tokensPerExpert[expert] = static_cast<std::size_t>(cache.hostAssignmentsPerExpert[expert]);
    }
    stats.routingEntropy = tokens == 0 ? 0.0 : hostSummary.entropySum / static_cast<double>(tokens);
    const std::size_t routed = slots - stats.droppedAssignments;
    double auxiliary = 0.0;
    if (routed > 0 && tokens > 0) {
        for (std::size_t expert = 0; expert < config_.numExperts; ++expert) {
            auxiliary += (static_cast<double>(stats.tokensPerExpert[expert]) / routed) *
                         (static_cast<double>(hostSummary.probabilityMass[expert]) / tokens);
        }
        auxiliary *= config_.numExperts;
    }
    stats.loadBalancingLoss = static_cast<float>(auxiliary);
    Tensor output = Tensor::zeros(input.shape(), MemoryArena::Activations);
    {
        // Le parentesi delimitano la fase al solo ciclo degli esperti:
        // senza, lo scope arriverebbe a fine funzione e si prenderebbe
        // anche il lavoro del router.
        const GpuPhaseScope expertScope(GpuPhase::Experts);
        for (std::size_t expert = 0; expert < config_.numExperts; ++expert) {
            const std::size_t count = static_cast<std::size_t>(cache.hostAssignmentsPerExpert[expert]);
            if (count == 0) continue;
            const int* assigned = cache.tokensOfExpert.data() + expert * cache.capacity;
            const int* expertSlots = cache.slotsOfExpert.data() + expert * cache.capacity;
            Tensor gathered({count, config_.hiddenSize}, MemoryArena::MoeDispatch);
            gatherRowsKernel<<<gridFor(gathered.elementCount()), kBlockSize>>>(
                gathered.data(), input.data(), assigned, count, config_.hiddenSize);
            Tensor expertOutput = experts_[expert].forward(gathered);
            combineRowsKernel<<<gridFor(expertOutput.elementCount()), kBlockSize>>>(
                output.data(), cache.expertOutputOfSlot.data(), expertOutput.data(), assigned, expertSlots,
                cache.weightOfSlot.data(), count, config_.hiddenSize);
            BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        }
    }
    return output;
}

Tensor MoELayer::backward(const Tensor& input, const Tensor& gradOutput, const MoECache& cache,
                          const MoERoutingStats& stats, GradientSink* sink) const {
    if (input.shape() != gradOutput.shape() || cache.tokens != input.elementCount() / config_.hiddenSize) {
        throw std::invalid_argument("CUDA MoELayer backward: shape/cache mismatch");
    }
    const std::size_t routed = stats.assignments - stats.droppedAssignments;
    Tensor gradProbabilities = Tensor::zeros({cache.tokens, config_.numExperts}, MemoryArena::MoeDispatch);
    GpuProfiler::instance().begin(GpuPhase::Router);
    routingGradientKernel<<<static_cast<unsigned int>(cache.tokens), 1>>>(
        gradProbabilities.data(), cache.probabilities.data(), cache.expertOfSlot.data(),
        cache.weightOfSlot.data(), cache.expertOutputOfSlot.data(), gradOutput.data(),
        cache.assignmentsPerExpert.data(), cache.tokens, config_.hiddenSize, config_.numExperts,
        config_.expertsPerToken, routed, config_.routerAuxLossWeight);
    GpuProfiler::instance().end(GpuPhase::Router);
    Tensor gradInput = Tensor::zeros(input.shape(), MemoryArena::Activations);
    {
        // Solo il ciclo: dopo di questo c'e' il backward del router, che
        // e' fase Router, non Experts.
        const GpuPhaseScope expertScope(GpuPhase::Experts);
        for (std::size_t expert = 0; expert < config_.numExperts; ++expert) {
            const std::size_t count = static_cast<std::size_t>(cache.hostAssignmentsPerExpert[expert]);
            if (count == 0) continue;
            const int* assigned = cache.tokensOfExpert.data() + expert * cache.capacity;
            const int* slots = cache.slotsOfExpert.data() + expert * cache.capacity;
            Tensor gatheredInput({count, config_.hiddenSize}, MemoryArena::MoeDispatch);
            Tensor gatheredGradient({count, config_.hiddenSize}, MemoryArena::MoeDispatch);
            gatherWeightedGradientKernel<<<gridFor(gatheredInput.elementCount()), kBlockSize>>>(
                gatheredInput.data(), gatheredGradient.data(), input.data(), gradOutput.data(), assigned,
                slots, cache.weightOfSlot.data(), count, config_.hiddenSize);
            Tensor expertGradInput = experts_[expert].backward(gatheredInput, gatheredGradient, sink);
            scatterAddKernel<<<gridFor(expertGradInput.elementCount()), kBlockSize>>>(
                gradInput.data(), expertGradInput.data(), assigned, count, config_.hiddenSize);
            BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        }
    }
    GpuProfiler::instance().begin(GpuPhase::Router);
    Tensor gradLogits({cache.tokens, config_.numExperts}, MemoryArena::MoeDispatch);
    softmaxBackwardSmallKernel<<<static_cast<unsigned int>(cache.tokens), 1>>>(
        gradLogits.data(), cache.probabilities.data(), gradProbabilities.data(), cache.tokens,
        config_.numExperts);
    DenseLinearGrad routerGrad = denseLinearBackward(input, routerWeight_, gradLogits);
    addInPlace(gradInput, routerGrad.input);
    if (sink != nullptr) {
        sink->consumeDenseGradient(routerParameterId(), routerGrad.weight.data(), routerGrad.weight.elementCount());
    }
    GpuProfiler::instance().end(GpuPhase::Router);
    return gradInput;
}

}  // namespace blackforge::blackbit::cuda
