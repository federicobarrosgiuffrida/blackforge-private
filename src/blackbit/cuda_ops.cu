#include "blackforge/blackbit/cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/rope.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;
constexpr float kRmsNormEpsilon = 1.0e-6F;

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void addKernel(float* output, const float* a, const float* b, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = a[i] + b[i];
}

__global__ void addInPlaceKernel(float* target, const float* source, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) target[i] += source[i];
}

__global__ void rmsNormForwardKernel(float* output, float* inverseRms, const float* input, const float* gamma,
                                     std::size_t rows, std::size_t features) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ double reduction[];
    double sum = 0.0;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        const double value = input[row * features + col];
        sum += value * value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inverseRms[row] = rsqrtf(static_cast<float>(reduction[0] / features) + kRmsNormEpsilon);
    __syncthreads();
    const float inverse = inverseRms[row];
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        output[row * features + col] = input[row * features + col] * inverse * gamma[col];
    }
}

__global__ void rmsNormBackwardKernel(float* gradInput, float* gradGamma, const float* input,
                                      const float* gamma, const float* gradOutput, const float* inverseRms,
                                      std::size_t rows, std::size_t features) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ double reduction[];
    double weighted = 0.0;
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        weighted += static_cast<double>(gradOutput[row * features + col]) * gamma[col] *
                    input[row * features + col];
    }
    reduction[threadIdx.x] = weighted;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = inverseRms[row];
    const float factor = static_cast<float>(reduction[0]) * inverse * inverse * inverse /
                         static_cast<float>(features);
    for (std::size_t col = threadIdx.x; col < features; col += blockDim.x) {
        const std::size_t index = row * features + col;
        gradInput[index] = gradOutput[index] * gamma[col] * inverse - factor * input[index];
        atomicAdd(gradGamma + col, gradOutput[index] * input[index] * inverse);
    }
}

__global__ void ropeKernel(float* tensor, std::size_t tokens, std::size_t seq, std::size_t heads,
                           std::size_t headDim, bool transpose) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pairs = headDim / 2;
    const std::size_t count = tokens * heads * pairs;
    if (index >= count) return;
    const std::size_t pair = index % pairs;
    const std::size_t head = (index / pairs) % heads;
    const std::size_t token = index / (pairs * heads);
    const std::size_t position = token % seq;
    float* values = tensor + (token * heads + head) * headDim;
    const float angle = ropeAngle(position, pair, headDim);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float a = values[pair];
    const float b = values[pair + pairs];
    if (transpose) {
        values[pair] = a * cosine + b * sine;
        values[pair + pairs] = -a * sine + b * cosine;
    } else {
        values[pair] = a * cosine - b * sine;
        values[pair + pairs] = a * sine + b * cosine;
    }
}

__device__ float siluValue(float value) { return value / (1.0F + expf(-value)); }

__global__ void siluMultiplyKernel(float* output, const float* gate, const float* up, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = siluValue(gate[i]) * up[i];
}

__global__ void siluMultiplyBackwardKernel(float* gradGate, float* gradUp, const float* gate,
                                           const float* up, const float* gradOutput, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float sigmoid = 1.0F / (1.0F + expf(-gate[i]));
    const float silu = gate[i] * sigmoid;
    gradUp[i] = gradOutput[i] * silu;
    gradGate[i] = gradOutput[i] * up[i] * sigmoid * (1.0F + gate[i] * (1.0F - sigmoid));
}

__global__ void embeddingKernel(float* output, const int* tokenIds, const std::uint32_t* packed,
                                const float* scales, std::size_t tokens, std::size_t vocab,
                                std::size_t hidden, std::size_t wordsPerRow, std::size_t groupsPerRow,
                                std::size_t groupSize, unsigned long long* invalid) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) return;
    const std::size_t token = index / hidden;
    const std::size_t col = index % hidden;
    const int id = tokenIds[token];
    if (id < 0 || static_cast<std::size_t>(id) >= vocab) {
        output[index] = 0.0F;
        atomicAdd(invalid, 1ULL);
        return;
    }
    const std::size_t row = static_cast<std::size_t>(id);
    const std::uint32_t word = packed[row * wordsPerRow + col / kTritsPerWord];
    const std::size_t inWord = col % kTritsPerWord;
    const int trit = decodeTritAt(wordByte(word, static_cast<int>(inWord / kTritsPerByte)),
                                  static_cast<int>(inWord % kTritsPerByte));
    output[index] = static_cast<float>(trit) * scales[row * groupsPerRow + col / groupSize];
}

struct LossStats {
    double sum;
    unsigned long long scored;
    unsigned long long nanInf;
    unsigned long long invalidTarget;
};

__global__ void crossEntropyKernel(float* gradient, const float* logits, const int* targets,
                                   std::size_t rows, std::size_t vocab, LossStats* stats) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float localMax = -3.402823466e+38F;
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
        localMax = fmaxf(localMax, logits[row * vocab + col]);
    }
    shared[threadIdx.x] = localMax;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = shared[0];
    float localSum = 0.0F;
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
        localSum += expf(logits[row * vocab + col] - maximum);
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = shared[0];
    const int target = targets[row];
    if (target < 0) {
        for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) gradient[row * vocab + col] = 0.0F;
        return;
    }
    if (static_cast<std::size_t>(target) >= vocab) {
        if (threadIdx.x == 0) atomicAdd(&stats->invalidTarget, 1ULL);
        return;
    }
    if (threadIdx.x == 0) {
        const float loss = maximum + logf(denominator) - logits[row * vocab + target];
        if (isfinite(loss)) atomicAdd(&stats->sum, static_cast<double>(loss));
        else atomicAdd(&stats->nanInf, 1ULL);
        atomicAdd(&stats->scored, 1ULL);
    }
    // Normalization by scored rows is applied in a second kernel.
    for (std::size_t col = threadIdx.x; col < vocab; col += blockDim.x) {
        float value = expf(logits[row * vocab + col] - maximum) / denominator;
        if (col == static_cast<std::size_t>(target)) value -= 1.0F;
        gradient[row * vocab + col] = value;
    }
}

__global__ void scaleKernel(float* values, std::size_t count, float factor) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) values[i] *= factor;
}

__global__ void denseLinearKernel(float* output, const float* input, const float* weight,
                                  std::size_t rows, std::size_t inFeatures, std::size_t outFeatures) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * outFeatures) return;
    const std::size_t row = index / outFeatures;
    const std::size_t out = index % outFeatures;
    float sum = 0.0F;
    for (std::size_t col = 0; col < inFeatures; ++col) sum += input[row * inFeatures + col] * weight[out * inFeatures + col];
    output[index] = sum;
}

__global__ void denseLinearBackwardKernel(float* gradInput, float* gradWeight, const float* input,
                                          const float* weight, const float* gradOutput, std::size_t rows,
                                          std::size_t inFeatures, std::size_t outFeatures) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * inFeatures) {
        const std::size_t row = index / inFeatures;
        const std::size_t col = index % inFeatures;
        float sum = 0.0F;
        for (std::size_t out = 0; out < outFeatures; ++out) sum += gradOutput[row * outFeatures + out] * weight[out * inFeatures + col];
        gradInput[index] = sum;
    }
    if (index < outFeatures * inFeatures) {
        const std::size_t out = index / inFeatures;
        const std::size_t col = index % inFeatures;
        float sum = 0.0F;
        for (std::size_t row = 0; row < rows; ++row) sum += gradOutput[row * outFeatures + out] * input[row * inFeatures + col];
        gradWeight[index] = sum;
    }
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b, MemoryArena arena) {
    if (a.shape() != b.shape()) throw std::invalid_argument("CUDA BlackBit add: shape mismatch");
    Tensor output(a.shape(), arena);
    addKernel<<<gridFor(a.elementCount()), kBlockSize>>>(output.data(), a.data(), b.data(), a.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return output;
}

void addInPlace(Tensor& target, const Tensor& source) {
    if (target.shape() != source.shape()) throw std::invalid_argument("CUDA BlackBit addInPlace: shape mismatch");
    addInPlaceKernel<<<gridFor(target.elementCount()), kBlockSize>>>(target.data(), source.data(), target.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
}

Tensor rmsNormForward(const Tensor& input, const Tensor& gamma, RmsNormCache& cache) {
    if (input.rank() < 2 || gamma.rank() != 1 || input.shape().back() != gamma.elementCount()) {
        throw std::invalid_argument("CUDA BlackBit RMSNorm: shape mismatch");
    }
    const std::size_t features = gamma.elementCount();
    const std::size_t rows = input.elementCount() / features;
    Tensor output(input.shape(), MemoryArena::Activations);
    cache.inverseRms = Tensor({rows}, MemoryArena::Activations);
    rmsNormForwardKernel<<<static_cast<unsigned int>(rows), kBlockSize, kBlockSize * sizeof(double)>>>(
        output.data(), cache.inverseRms.data(), input.data(), gamma.data(), rows, features);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return output;
}

RmsNormGrad rmsNormBackward(const Tensor& input, const Tensor& gamma, const Tensor& gradOutput,
                            const RmsNormCache& cache) {
    if (input.shape() != gradOutput.shape()) throw std::invalid_argument("CUDA BlackBit RMSNorm backward: shape mismatch");
    const std::size_t features = gamma.elementCount();
    const std::size_t rows = input.elementCount() / features;
    RmsNormGrad result{Tensor(input.shape(), MemoryArena::Activations),
                       Tensor::zeros({features}, MemoryArena::GradientTiles)};
    rmsNormBackwardKernel<<<static_cast<unsigned int>(rows), kBlockSize, kBlockSize * sizeof(double)>>>(
        result.input.data(), result.gamma.data(), input.data(), gamma.data(), gradOutput.data(),
        cache.inverseRms.data(), rows, features);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return result;
}

void applyRopeInPlace(Tensor& tensor, std::size_t batch, std::size_t seq, std::size_t heads,
                      std::size_t headDim, bool transpose) {
    if (headDim % 2 != 0 || tensor.elementCount() != batch * seq * heads * headDim) {
        throw std::invalid_argument("CUDA BlackBit RoPE: shape mismatch");
    }
    const std::size_t count = batch * seq * heads * (headDim / 2);
    ropeKernel<<<gridFor(count), kBlockSize>>>(tensor.data(), batch * seq, seq, heads, headDim, transpose);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
}

Tensor siluMultiply(const Tensor& gate, const Tensor& up) {
    if (gate.shape() != up.shape()) throw std::invalid_argument("CUDA BlackBit SwiGLU: shape mismatch");
    Tensor output(gate.shape(), MemoryArena::Activations);
    siluMultiplyKernel<<<gridFor(gate.elementCount()), kBlockSize>>>(output.data(), gate.data(), up.data(), gate.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return output;
}

SiluMultiplyGrad siluMultiplyBackward(const Tensor& gate, const Tensor& up, const Tensor& gradOutput) {
    if (gate.shape() != up.shape() || gate.shape() != gradOutput.shape()) throw std::invalid_argument("CUDA BlackBit SwiGLU backward: shape mismatch");
    SiluMultiplyGrad result{Tensor(gate.shape(), MemoryArena::Activations), Tensor(up.shape(), MemoryArena::Activations)};
    siluMultiplyBackwardKernel<<<gridFor(gate.elementCount()), kBlockSize>>>(
        result.gate.data(), result.up.data(), gate.data(), up.data(), gradOutput.data(), gate.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor embeddingLookup(const IndexTensor& tokenIds, const cuda::TernaryTensor& table, std::size_t hidden) {
    if (table.rowLength() != hidden) throw std::invalid_argument("CUDA BlackBit embedding: hidden size mismatch");
    std::vector<std::size_t> shape = tokenIds.shape();
    shape.push_back(hidden);
    Tensor output(std::move(shape), MemoryArena::Activations);
    Buffer invalid(sizeof(unsigned long long), MemoryArena::Temporary);
    BLACKFORGE_CUDA_CHECK(cudaMemset(invalid.data(), 0, invalid.bytes()));
    embeddingKernel<<<gridFor(output.elementCount()), kBlockSize>>>(
        output.data(), tokenIds.data(), table.packedWords(), table.scales(), tokenIds.elementCount(), table.rows(),
        hidden, table.wordsPerRow(), table.groupsPerRow(), table.groupSize(), invalid.as<unsigned long long>());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    unsigned long long invalidCount = 0;
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(&invalidCount, invalid.data(), sizeof(invalidCount), cudaMemcpyDeviceToHost));
    if (invalidCount != 0) throw std::invalid_argument("CUDA BlackBit embedding: token id outside vocabulary");
    return output;
}

CrossEntropyResult crossEntropySparse(const Tensor& logits, const IndexTensor& targets) {
    if (logits.rank() != 2 || targets.elementCount() != logits.dim(0)) throw std::invalid_argument("CUDA BlackBit cross entropy: shape mismatch");
    CrossEntropyResult result;
    result.gradient = Tensor(logits.shape(), MemoryArena::GradientTiles);
    Buffer stats(sizeof(LossStats), MemoryArena::Temporary);
    BLACKFORGE_CUDA_CHECK(cudaMemset(stats.data(), 0, stats.bytes()));
    crossEntropyKernel<<<static_cast<unsigned int>(logits.dim(0)), kBlockSize, kBlockSize * sizeof(float)>>>(
        result.gradient.data(), logits.data(), targets.data(), logits.dim(0), logits.dim(1), stats.as<LossStats>());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    LossStats host{};
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(&host, stats.data(), sizeof(host), cudaMemcpyDeviceToHost));
    if (host.invalidTarget != 0) throw std::invalid_argument("CUDA BlackBit cross entropy: target outside vocabulary");
    result.scoredTokens = static_cast<std::size_t>(host.scored);
    result.nanInfCount = static_cast<std::size_t>(host.nanInf);
    result.loss = host.scored == 0 ? 0.0F : static_cast<float>(host.sum / host.scored);
    if (host.scored != 0) {
        scaleKernel<<<gridFor(result.gradient.elementCount()), kBlockSize>>>(
            result.gradient.data(), result.gradient.elementCount(), 1.0F / static_cast<float>(host.scored));
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

Tensor denseLinear(const Tensor& input, const Tensor& weight) {
    if (input.rank() < 2 || weight.rank() != 2 || input.shape().back() != weight.dim(1)) throw std::invalid_argument("CUDA BlackBit denseLinear: shape mismatch");
    const std::size_t rows = input.elementCount() / weight.dim(1);
    std::vector<std::size_t> shape = input.shape();
    shape.back() = weight.dim(0);
    Tensor output(std::move(shape), MemoryArena::Activations);
    denseLinearKernel<<<gridFor(output.elementCount()), kBlockSize>>>(output.data(), input.data(), weight.data(), rows, weight.dim(1), weight.dim(0));
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return output;
}

DenseLinearGrad denseLinearBackward(const Tensor& input, const Tensor& weight, const Tensor& gradOutput) {
    if (input.rank() < 2 || weight.rank() != 2 || gradOutput.shape().back() != weight.dim(0)) throw std::invalid_argument("CUDA BlackBit denseLinear backward: shape mismatch");
    const std::size_t rows = input.elementCount() / weight.dim(1);
    DenseLinearGrad result{Tensor(input.shape(), MemoryArena::Activations), Tensor(weight.shape(), MemoryArena::GradientTiles)};
    const std::size_t threads = std::max(result.input.elementCount(), result.weight.elementCount());
    denseLinearBackwardKernel<<<gridFor(threads), kBlockSize>>>(result.input.data(), result.weight.data(), input.data(), weight.data(), gradOutput.data(), rows, weight.dim(1), weight.dim(0));
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return result;
}

}  // namespace blackforge::blackbit::cuda
