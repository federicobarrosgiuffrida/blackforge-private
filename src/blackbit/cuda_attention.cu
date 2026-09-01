#include "blackforge/blackbit/cuda_attention.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kAttentionThreads = 256;
constexpr int kHeadDim128Threads = 128;
constexpr int kCudaWarpSize = 32;

__device__ __forceinline__ float warpSum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <int Threads>
__device__ __forceinline__ float blockSum(float value, float* warpSums) {
    static_assert(Threads % kCudaWarpSize == 0);
    constexpr int warps = Threads / kCudaWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kCudaWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kCudaWarpSize;
    value = warpSum(value);
    if (lane == 0) warpSums[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < warps ? warpSums[lane] : 0.0F;
        value = warpSum(value);
        if (lane == 0) warpSums[0] = value;
    }
    __syncthreads();
    return warpSums[0];
}

template <int Threads>
__device__ __forceinline__ void blockSumPair(float& first, float& second,
                                              float* firstWarpSums, float* secondWarpSums) {
    static_assert(Threads % kCudaWarpSize == 0);
    constexpr int warps = Threads / kCudaWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kCudaWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kCudaWarpSize;
    first = warpSum(first);
    second = warpSum(second);
    if (lane == 0) {
        firstWarpSums[warp] = first;
        secondWarpSums[warp] = second;
    }
    __syncthreads();
    if (warp == 0) {
        first = lane < warps ? firstWarpSums[lane] : 0.0F;
        second = lane < warps ? secondWarpSums[lane] : 0.0F;
        first = warpSum(first);
        second = warpSum(second);
        if (lane == 0) {
            firstWarpSums[0] = first;
            secondWarpSums[0] = second;
        }
    }
    __syncthreads();
    first = firstWarpSums[0];
    second = secondWarpSums[0];
}

// The production BlackBit models use head_dim=128. One thread owns one head
// element and warp shuffles replace the eight block-wide reduction barriers in
// the generic reference kernel.
__global__ __launch_bounds__(kHeadDim128Threads) void gqaForward128Kernel(
    float* output, float* rowMax, float* rowSum, const float* query, const float* key,
    const float* value, std::size_t batch, std::size_t seq, std::size_t heads,
    std::size_t kvHeads, std::size_t headsPerGroup, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    if (batchIndex >= batch) return;
    const std::size_t kvHead = head / headsPerGroup;
    const std::size_t dimension = threadIdx.x;
    __shared__ float warpSums[kHeadDim128Threads / kCudaWarpSize];
    __shared__ float correction;
    __shared__ float weight;
    __shared__ float runningMaximum;
    __shared__ float runningTotal;
    if (dimension == 0) {
        runningMaximum = -3.402823466e+38F;
        runningTotal = 0.0F;
    }
    __syncthreads();
    const std::size_t queryOffset = ((batchIndex * seq + queryPosition) * heads + head) * 128;
    const float queryValue = query[queryOffset + dimension];
    float accumulator = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const std::size_t kvOffset = ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * 128;
        const float scoreSum = blockSum<kHeadDim128Threads>(queryValue * key[kvOffset + dimension], warpSums);
        if (dimension == 0) {
            const float score = scoreSum * scale;
            const float nextMaximum = fmaxf(runningMaximum, score);
            correction = runningMaximum == -3.402823466e+38F ? 0.0F : expf(runningMaximum - nextMaximum);
            weight = expf(score - nextMaximum);
            runningTotal = fmaf(runningTotal, correction, weight);
            runningMaximum = nextMaximum;
        }
        __syncthreads();
        accumulator = fmaf(weight, value[kvOffset + dimension], accumulator * correction);
    }
    output[queryOffset + dimension] = accumulator / runningTotal;
    if (dimension == 0) {
        rowMax[row] = runningMaximum;
        rowSum[row] = runningTotal;
    }
}

// Parametrico in kHeadDim: ogni lane possiede kHeadDim/32 dimensioni, quindi
// 2 per headDim 64 e 4 per headDim 128. Il modello 9B usa 128, i modelli
// piccoli 64, e prima solo il primo prendeva questa strada.
//
// Il kernel per riga singola spende, per OGNI coppia (query, chiave), una
// riduzione di blocco su 128 thread: due __syncthreads e un giro in memoria
// condivisa per un solo prodotto scalare. Con seq=512 sono 2,5e7 riduzioni
// per layer, ed e' li' che se ne va il tempo, non nei calcoli.
//
// Qui una riga di query e' posseduta da un WARP: ogni lane tiene 4 delle 128
// dimensioni (lane, lane+32, lane+64, lane+96), il prodotto scalare si chiude
// con una farfalla di __shfl_xor_sync che lascia la somma in tutte le lane, e
// gli accumulatori su V restano nei registri della lane che li possiede.
// Niente barriere, niente memoria condivisa: i 4 warp del blocco avanzano
// indipendenti e coprono 4 righe di query invece di una.
//
// Gli accessi restano coalescenti: le lane 0..31 leggono indirizzi
// consecutivi per ciascuno dei 4 blocchi di dimensioni.
//
// La ricorrenza del softmax online e' identica; cambia solo l'ordine di
// somma dentro il prodotto scalare, quindi il risultato e' equivalente entro
// la tolleranza in virgola mobile, non bit-identico.
__device__ __forceinline__ float warpAllSum(float value) {
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

constexpr int kRowsPerWarpBlock = 4;
constexpr int kWarpBlockThreads = kRowsPerWarpBlock * kCudaWarpSize;

template <int kHeadDim>
__global__ __launch_bounds__(kWarpBlockThreads) void gqaForwardWarpKernel(
    float* output, float* rowMax, float* rowSum, const float* query, const float* key,
    const float* value, std::size_t batch, std::size_t seq, std::size_t heads,
    std::size_t kvHeads, std::size_t headsPerGroup, float scale) {
    const int warp = static_cast<int>(threadIdx.x) / kCudaWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kCudaWarpSize;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * kRowsPerWarpBlock + static_cast<std::size_t>(warp);
    if (row >= batch * heads * seq) return;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    const std::size_t kvHead = head / headsPerGroup;
    const std::size_t queryOffset = ((batchIndex * seq + queryPosition) * heads + head) * kHeadDim;

    float queryValue[kHeadDim / kCudaWarpSize];
    float accumulator[kHeadDim / kCudaWarpSize];
#pragma unroll
    for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
        queryValue[i] = query[queryOffset + lane + i * kCudaWarpSize];
        accumulator[i] = 0.0F;
    }

    float runningMaximum = -3.402823466e+38F;
    float runningTotal = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const std::size_t kvOffset = ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * kHeadDim;
        float partial = 0.0F;
        float valueValue[kHeadDim / kCudaWarpSize];
#pragma unroll
        for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
            partial = fmaf(queryValue[i], key[kvOffset + lane + i * kCudaWarpSize], partial);
            valueValue[i] = value[kvOffset + lane + i * kCudaWarpSize];
        }
        const float score = warpAllSum(partial) * scale;
        const float nextMaximum = fmaxf(runningMaximum, score);
        const float correction =
            runningMaximum == -3.402823466e+38F ? 0.0F : expf(runningMaximum - nextMaximum);
        const float weight = expf(score - nextMaximum);
        runningTotal = fmaf(runningTotal, correction, weight);
        runningMaximum = nextMaximum;
#pragma unroll
        for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
            accumulator[i] = fmaf(weight, valueValue[i], accumulator[i] * correction);
        }
    }

#pragma unroll
    for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
        output[queryOffset + lane + i * kCudaWarpSize] = accumulator[i] / runningTotal;
    }
    if (lane == 0) {
        rowMax[row] = runningMaximum;
        rowSum[row] = runningTotal;
    }
}

__global__ __launch_bounds__(kHeadDim128Threads) void gqaBackward128Kernel(
    float* gradQuery, float* gradKey, float* gradValue, const float* query, const float* key,
    const float* value, const float* output, const float* gradOutput, const float* rowMax,
    const float* rowSum, std::size_t batch, std::size_t seq, std::size_t heads,
    std::size_t kvHeads, std::size_t headsPerGroup, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    if (batchIndex >= batch) return;
    const std::size_t kvHead = head / headsPerGroup;
    const std::size_t dimension = threadIdx.x;
    const std::size_t queryOffset = ((batchIndex * seq + queryPosition) * heads + head) * 128;
    const float queryValue = query[queryOffset + dimension];
    const float gradOutputValue = gradOutput[queryOffset + dimension];
    __shared__ float dotWarpSums[kHeadDim128Threads / kCudaWarpSize];
    __shared__ float gradientWarpSums[kHeadDim128Threads / kCudaWarpSize];
    __shared__ float probability;
    __shared__ float gradScore;
    const float correctionValue = blockSum<kHeadDim128Threads>(
        output[queryOffset + dimension] * gradOutputValue, gradientWarpSums);
    float queryGradient = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const std::size_t kvOffset = ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * 128;
        const float keyValue = key[kvOffset + dimension];
        const float valueValue = value[kvOffset + dimension];
        float dot = queryValue * keyValue;
        float probabilityGradient = gradOutputValue * valueValue;
        blockSumPair<kHeadDim128Threads>(dot, probabilityGradient, dotWarpSums, gradientWarpSums);
        if (dimension == 0) {
            probability = expf(dot * scale - rowMax[row]) / rowSum[row];
            gradScore = probability * (probabilityGradient - correctionValue) * scale;
        }
        __syncthreads();
        queryGradient = fmaf(gradScore, keyValue, queryGradient);
        atomicAdd(gradKey + kvOffset + dimension, gradScore * queryValue);
        atomicAdd(gradValue + kvOffset + dimension, probability * gradOutputValue);
    }
    gradQuery[queryOffset + dimension] = queryGradient;
}

__global__ void gqaForwardKernel(float* output, float* rowMax, float* rowSum, const float* query,
                                 const float* key, const float* value, std::size_t batch,
                                 std::size_t seq, std::size_t heads, std::size_t kvHeads,
                                 std::size_t headDim, std::size_t headsPerGroup, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    if (batchIndex >= batch) return;
    const std::size_t kvHead = head / headsPerGroup;
    __shared__ double reduction[kAttentionThreads];
    __shared__ float correction;
    __shared__ float weight;
    __shared__ float runningMaximum;
    __shared__ float runningTotal;
    if (threadIdx.x == 0) {
        runningMaximum = -3.402823466e+38F;
        runningTotal = 0.0F;
    }
    __syncthreads();
    const float* queryRow = query + ((batchIndex * seq + queryPosition) * heads + head) * headDim;
    float accumulator = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const float* keyRow = key + ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * headDim;
        double local = 0.0;
        for (std::size_t dimension = threadIdx.x; dimension < headDim; dimension += blockDim.x) {
            local += static_cast<double>(queryRow[dimension]) * static_cast<double>(keyRow[dimension]);
        }
        reduction[threadIdx.x] = local;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = static_cast<float>(reduction[0]) * scale;
            const float nextMaximum = fmaxf(runningMaximum, score);
            correction = runningMaximum == -3.402823466e+38F ? 0.0F : expf(runningMaximum - nextMaximum);
            weight = expf(score - nextMaximum);
            runningTotal = runningTotal * correction + weight;
            runningMaximum = nextMaximum;
        }
        __syncthreads();
        if (threadIdx.x < headDim) {
            const float* valueRow = value + ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * headDim;
            accumulator = accumulator * correction + weight * valueRow[threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x < headDim) {
        output[((batchIndex * seq + queryPosition) * heads + head) * headDim + threadIdx.x] =
            accumulator / runningTotal;
    }
    if (threadIdx.x == 0) {
        rowMax[row] = runningMaximum;
        rowSum[row] = runningTotal;
    }
}

// Stessa trasformazione del forward applicata al backward: una riga di query
// per warp, 4 dimensioni per lane, riduzioni con farfalla invece che con
// riduzione di blocco. Qui le riduzioni per chiave sono due (il prodotto
// scalare e il gradiente sulla probabilita'), quindi il vecchio schema
// pagava due __syncthreads e due giri in memoria condivisa per ogni coppia.
//
// Gli atomicAdd su gradKey e gradValue restano necessari perche' piu' righe
// di query concorrono sulla stessa chiave, ma ora le lane di un warp
// scrivono indirizzi consecutivi, quindi le transazioni sono coalescenti.
__device__ __forceinline__ void warpAllSumPair(float& first, float& second) {
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        first += __shfl_xor_sync(0xffffffffU, first, offset);
        second += __shfl_xor_sync(0xffffffffU, second, offset);
    }
}

template <int kHeadDim>
__global__ __launch_bounds__(kWarpBlockThreads) void gqaBackwardWarpKernel(
    float* gradQuery, float* gradKey, float* gradValue, const float* query, const float* key,
    const float* value, const float* output, const float* gradOutput, const float* rowMax,
    const float* rowSum, std::size_t batch, std::size_t seq, std::size_t heads,
    std::size_t kvHeads, std::size_t headsPerGroup, float scale) {
    const int warp = static_cast<int>(threadIdx.x) / kCudaWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kCudaWarpSize;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * kRowsPerWarpBlock + static_cast<std::size_t>(warp);
    if (row >= batch * heads * seq) return;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    const std::size_t kvHead = head / headsPerGroup;
    const std::size_t queryOffset = ((batchIndex * seq + queryPosition) * heads + head) * kHeadDim;

    float queryValue[kHeadDim / kCudaWarpSize];
    float gradOutputValue[kHeadDim / kCudaWarpSize];
    float queryGradient[kHeadDim / kCudaWarpSize];
    float correctionPartial = 0.0F;
#pragma unroll
    for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
        const std::size_t offset = queryOffset + lane + i * kCudaWarpSize;
        queryValue[i] = query[offset];
        gradOutputValue[i] = gradOutput[offset];
        queryGradient[i] = 0.0F;
        correctionPartial = fmaf(output[offset], gradOutputValue[i], correctionPartial);
    }
    const float correctionValue = warpAllSum(correctionPartial);
    const float rowMaximum = rowMax[row];
    const float rowTotal = rowSum[row];

    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const std::size_t kvOffset = ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * kHeadDim;
        float keyValue[kHeadDim / kCudaWarpSize];
        float valueValue[kHeadDim / kCudaWarpSize];
        float dot = 0.0F;
        float probabilityGradient = 0.0F;
#pragma unroll
        for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
            const std::size_t offset = kvOffset + lane + i * kCudaWarpSize;
            keyValue[i] = key[offset];
            valueValue[i] = value[offset];
            dot = fmaf(queryValue[i], keyValue[i], dot);
            probabilityGradient = fmaf(gradOutputValue[i], valueValue[i], probabilityGradient);
        }
        warpAllSumPair(dot, probabilityGradient);
        const float probability = expf(dot * scale - rowMaximum) / rowTotal;
        const float gradScore = probability * (probabilityGradient - correctionValue) * scale;
#pragma unroll
        for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
            const std::size_t offset = kvOffset + lane + i * kCudaWarpSize;
            queryGradient[i] = fmaf(gradScore, keyValue[i], queryGradient[i]);
            atomicAdd(gradKey + offset, gradScore * queryValue[i]);
            atomicAdd(gradValue + offset, probability * gradOutputValue[i]);
        }
    }

#pragma unroll
    for (int i = 0; i < kHeadDim / kCudaWarpSize; ++i) {
        gradQuery[queryOffset + lane + i * kCudaWarpSize] = queryGradient[i];
    }
}

__global__ void gqaBackwardKernel(float* gradQuery, float* gradKey, float* gradValue,
                                  const float* query, const float* key, const float* value,
                                  const float* output, const float* gradOutput, const float* rowMax,
                                  const float* rowSum, std::size_t batch, std::size_t seq,
                                  std::size_t heads, std::size_t kvHeads, std::size_t headDim,
                                  std::size_t headsPerGroup, float scale) {
    const std::size_t row = blockIdx.x;
    const std::size_t queryPosition = row % seq;
    const std::size_t head = (row / seq) % heads;
    const std::size_t batchIndex = row / (seq * heads);
    if (batchIndex >= batch) return;
    const std::size_t kvHead = head / headsPerGroup;
    __shared__ double dotReduction[kAttentionThreads];
    __shared__ double probabilityReduction[kAttentionThreads];
    __shared__ float probability;
    __shared__ float gradScore;
    const std::size_t queryOffset = ((batchIndex * seq + queryPosition) * heads + head) * headDim;
    const float* queryRow = query + queryOffset;
    const float* outputRow = output + queryOffset;
    const float* gradOutputRow = gradOutput + queryOffset;
    double localCorrection = 0.0;
    for (std::size_t dimension = threadIdx.x; dimension < headDim; dimension += blockDim.x) {
        localCorrection += static_cast<double>(outputRow[dimension]) * gradOutputRow[dimension];
    }
    probabilityReduction[threadIdx.x] = localCorrection;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) probabilityReduction[threadIdx.x] += probabilityReduction[threadIdx.x + stride];
        __syncthreads();
    }
    const double correctionValue = probabilityReduction[0];
    float queryGradient = 0.0F;
    for (std::size_t keyPosition = 0; keyPosition <= queryPosition; ++keyPosition) {
        const std::size_t kvOffset = ((batchIndex * seq + keyPosition) * kvHeads + kvHead) * headDim;
        const float* keyRow = key + kvOffset;
        const float* valueRow = value + kvOffset;
        double localDot = 0.0;
        double localProbabilityGradient = 0.0;
        for (std::size_t dimension = threadIdx.x; dimension < headDim; dimension += blockDim.x) {
            localDot += static_cast<double>(queryRow[dimension]) * keyRow[dimension];
            localProbabilityGradient += static_cast<double>(gradOutputRow[dimension]) * valueRow[dimension];
        }
        dotReduction[threadIdx.x] = localDot;
        probabilityReduction[threadIdx.x] = localProbabilityGradient;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                dotReduction[threadIdx.x] += dotReduction[threadIdx.x + stride];
                probabilityReduction[threadIdx.x] += probabilityReduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            probability = expf(static_cast<float>(dotReduction[0]) * scale - rowMax[row]) / rowSum[row];
            gradScore = probability * static_cast<float>(probabilityReduction[0] - correctionValue) * scale;
        }
        __syncthreads();
        if (threadIdx.x < headDim) {
            queryGradient += gradScore * keyRow[threadIdx.x];
            atomicAdd(gradKey + kvOffset + threadIdx.x, gradScore * queryRow[threadIdx.x]);
            atomicAdd(gradValue + kvOffset + threadIdx.x, probability * gradOutputRow[threadIdx.x]);
        }
        __syncthreads();
    }
    if (threadIdx.x < headDim) gradQuery[queryOffset + threadIdx.x] = queryGradient;
}

}  // namespace

GqaAttention::GqaAttention(blackforge::blackbit::GqaAttention& cpuReference, const BlackBitConfig& config)
    : config_(config),
      q_(cpuReference.queryProjection()),
      k_(cpuReference.keyProjection()),
      v_(cpuReference.valueProjection()),
      o_(cpuReference.outputProjection()) {
    config_.validate();
    if (config_.headDim > kAttentionThreads || config_.headDim % 2 != 0) {
        throw std::invalid_argument("CUDA GqaAttention: head_dim must be even and at most 256");
    }
}

Tensor GqaAttention::forward(const Tensor& input, AttentionCache& cache) const {
    if (input.rank() != 3 || input.dim(2) != config_.hiddenSize) {
        throw std::invalid_argument("CUDA GqaAttention: expected [batch, seq, hidden]");
    }
    const std::size_t batch = input.dim(0);
    const std::size_t seq = input.dim(1);
    if (seq > config_.maxSeqLen) throw std::invalid_argument("CUDA GqaAttention: sequence exceeds max_seq_len");
    cache.query = q_.forward(input);
    cache.key = k_.forward(input);
    cache.value = v_.forward(input);
    applyRopeInPlace(cache.query, batch, seq, config_.numHeads, config_.headDim);
    applyRopeInPlace(cache.key, batch, seq, config_.numKvHeads, config_.headDim);
    cache.attnOutput = Tensor({batch, seq, config_.queryDim()}, MemoryArena::Attention);
    cache.rowMax = Tensor({batch, config_.numHeads, seq}, MemoryArena::Attention);
    cache.rowSum = Tensor({batch, config_.numHeads, seq}, MemoryArena::Attention);
    const std::size_t rows = batch * config_.numHeads * seq;
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.headDim));
    const std::size_t warpBlocks = (rows + kRowsPerWarpBlock - 1) / kRowsPerWarpBlock;
    if (config_.headDim == 128) {
        gqaForwardWarpKernel<128><<<static_cast<unsigned int>(warpBlocks), kWarpBlockThreads>>>(
            cache.attnOutput.data(), cache.rowMax.data(), cache.rowSum.data(), cache.query.data(),
            cache.key.data(), cache.value.data(), batch, seq, config_.numHeads, config_.numKvHeads,
            config_.headsPerKvGroup(), scale);
    } else if (config_.headDim == 64) {
        gqaForwardWarpKernel<64><<<static_cast<unsigned int>(warpBlocks), kWarpBlockThreads>>>(
            cache.attnOutput.data(), cache.rowMax.data(), cache.rowSum.data(), cache.query.data(),
            cache.key.data(), cache.value.data(), batch, seq, config_.numHeads, config_.numKvHeads,
            config_.headsPerKvGroup(), scale);
    } else {
        gqaForwardKernel<<<static_cast<unsigned int>(rows), kAttentionThreads>>>(
            cache.attnOutput.data(), cache.rowMax.data(), cache.rowSum.data(), cache.query.data(),
            cache.key.data(), cache.value.data(), batch, seq, config_.numHeads, config_.numKvHeads,
            config_.headDim, config_.headsPerKvGroup(), scale);
    }
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return o_.forward(cache.attnOutput);
}

Tensor GqaAttention::backward(const Tensor& input, const Tensor& gradOutput, const AttentionCache& cache,
                              GradientSink* sink) const {
    const std::size_t batch = input.dim(0);
    const std::size_t seq = input.dim(1);
    Tensor gradAttnOutput = o_.backward(cache.attnOutput, gradOutput, sink);
    Tensor gradQuery = Tensor::zeros(cache.query.shape(), MemoryArena::Attention);
    Tensor gradKey = Tensor::zeros(cache.key.shape(), MemoryArena::Attention);
    Tensor gradValue = Tensor::zeros(cache.value.shape(), MemoryArena::Attention);
    const std::size_t rows = batch * config_.numHeads * seq;
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.headDim));
    const std::size_t warpBlocks = (rows + kRowsPerWarpBlock - 1) / kRowsPerWarpBlock;
    if (config_.headDim == 128) {
        gqaBackwardWarpKernel<128><<<static_cast<unsigned int>(warpBlocks), kWarpBlockThreads>>>(
            gradQuery.data(), gradKey.data(), gradValue.data(), cache.query.data(), cache.key.data(),
            cache.value.data(), cache.attnOutput.data(), gradAttnOutput.data(), cache.rowMax.data(),
            cache.rowSum.data(), batch, seq, config_.numHeads, config_.numKvHeads,
            config_.headsPerKvGroup(), scale);
    } else if (config_.headDim == 64) {
        gqaBackwardWarpKernel<64><<<static_cast<unsigned int>(warpBlocks), kWarpBlockThreads>>>(
            gradQuery.data(), gradKey.data(), gradValue.data(), cache.query.data(), cache.key.data(),
            cache.value.data(), cache.attnOutput.data(), gradAttnOutput.data(), cache.rowMax.data(),
            cache.rowSum.data(), batch, seq, config_.numHeads, config_.numKvHeads,
            config_.headsPerKvGroup(), scale);
    } else {
        gqaBackwardKernel<<<static_cast<unsigned int>(rows), kAttentionThreads>>>(
            gradQuery.data(), gradKey.data(), gradValue.data(), cache.query.data(), cache.key.data(),
            cache.value.data(), cache.attnOutput.data(), gradAttnOutput.data(), cache.rowMax.data(),
            cache.rowSum.data(), batch, seq, config_.numHeads, config_.numKvHeads, config_.headDim,
            config_.headsPerKvGroup(), scale);
    }
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    applyRopeInPlace(gradQuery, batch, seq, config_.numHeads, config_.headDim, true);
    applyRopeInPlace(gradKey, batch, seq, config_.numKvHeads, config_.headDim, true);
    Tensor fromQuery = q_.backward(input, gradQuery, sink);
    Tensor fromKey = k_.backward(input, gradKey, sink);
    Tensor fromValue = v_.backward(input, gradValue, sink);
    Tensor result = add(fromQuery, fromKey, MemoryArena::Activations);
    addInPlace(result, fromValue);
    return result;
}

}  // namespace blackforge::blackbit::cuda
