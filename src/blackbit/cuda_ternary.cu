#include "blackforge/blackbit/cuda_ternary.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;

std::size_t productExceptLast(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end() - 1, static_cast<std::size_t>(1), std::multiplies<>());
}

__global__ void packTritsKernel(std::uint32_t* packed, const std::int8_t* trits, std::size_t rows,
                                std::size_t rowLength, std::size_t wordsPerRow) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t wordCount = rows * wordsPerRow;
    if (index >= wordCount) {
        return;
    }
    const std::size_t row = index / wordsPerRow;
    const std::size_t firstCol = (index % wordsPerRow) * kTritsPerWord;
    std::uint32_t word = 0;
    for (int byte = 0; byte < 4; ++byte) {
        int values[kTritsPerByte] = {0, 0, 0, 0, 0};
        for (std::size_t slot = 0; slot < kTritsPerByte; ++slot) {
            const std::size_t col = firstCol + static_cast<std::size_t>(byte) * kTritsPerByte + slot;
            if (col < rowLength) {
                values[slot] = static_cast<int>(trits[row * rowLength + col]);
            }
        }
        word = setWordByte(word, byte,
                           encodeTritByte(values[0], values[1], values[2], values[3], values[4]));
    }
    packed[index] = word;
}

__global__ void decodeRowsKernel(float* output, const std::uint32_t* packed, const float* scales,
                                  std::size_t firstRow, std::size_t rowCount, std::size_t rowLength,
                                  std::size_t wordsPerRow, std::size_t groupsPerRow, std::size_t groupSize) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rowCount * rowLength;
    if (index >= count) {
        return;
    }
    const std::size_t localRow = index / rowLength;
    const std::size_t col = index % rowLength;
    const std::size_t row = firstRow + localRow;
    const std::uint32_t word = packed[row * wordsPerRow + col / kTritsPerWord];
    const std::size_t inWord = col % kTritsPerWord;
    const int trit = decodeTritAt(wordByte(word, static_cast<int>(inWord / kTritsPerByte)),
                                  static_cast<int>(inWord % kTritsPerByte));
    output[index] = static_cast<float>(trit) * scales[row * groupsPerRow + col / groupSize];
}

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

DecodedTile::DecodedTile(std::size_t rows, std::size_t cols)
    : storage_(rows * cols * sizeof(float), MemoryArena::DequantizationTiles), rows_(rows), cols_(cols) {}

TernaryTensor::TernaryTensor(std::vector<std::size_t> shape, std::size_t groupSize)
    : shape_(std::move(shape)), groupSize_(groupSize) {
    if (shape_.empty() || shape_.back() == 0 || groupSize_ == 0 || groupSize_ % kTritsPerWord != 0) {
        throw std::invalid_argument("CUDA TernaryTensor: invalid shape or group size");
    }
    rows_ = productExceptLast(shape_);
    rowLength_ = shape_.back();
    if (rows_ == 0) {
        throw std::invalid_argument("CUDA TernaryTensor: dimensions must be non-zero");
    }
    wordsPerRow_ = ternaryWordsPerRow(rowLength_);
    groupsPerRow_ = (rowLength_ + groupSize_ - 1) / groupSize_;
    allocate();
}

void TernaryTensor::allocate() {
    packed_ = Buffer(rows_ * wordsPerRow_ * sizeof(std::uint32_t), MemoryArena::PackedWeights);
    scales_ = Buffer(rows_ * groupsPerRow_ * sizeof(float), MemoryArena::Scales);
}

TernaryTensor::TernaryTensor(const blackforge::blackbit::TernaryTensor& host)
    : TernaryTensor(host.shape(), host.groupSize()) {
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(packed_.data(), host.packedWords().data(), packed_.bytes(), cudaMemcpyHostToDevice));
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(scales_.data(), host.scales().data(), scales_.bytes(), cudaMemcpyHostToDevice));
}

TernaryTensor TernaryTensor::fromTrits(std::vector<std::size_t> shape, const std::vector<std::int8_t>& trits,
                                        const std::vector<float>& scales, std::size_t groupSize) {
    TernaryTensor result(std::move(shape), groupSize);
    if (trits.size() != result.elementCount() || scales.size() != result.rows_ * result.groupsPerRow_) {
        throw std::invalid_argument("CUDA TernaryTensor::fromTrits: logical data does not match shape");
    }
    if (std::any_of(trits.begin(), trits.end(), [](std::int8_t value) { return value < -1 || value > 1; })) {
        throw std::invalid_argument("CUDA TernaryTensor::fromTrits: trits must be -1, 0, or +1");
    }

    Buffer logical(trits.size() * sizeof(std::int8_t), MemoryArena::Temporary);
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(logical.data(), trits.data(), logical.bytes(), cudaMemcpyHostToDevice));
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(result.scales_.data(), scales.data(), result.scales_.bytes(),
                                     cudaMemcpyHostToDevice));

    const std::size_t words = result.rows_ * result.wordsPerRow_;
    packTritsKernel<<<gridFor(words), kBlockSize>>>(result.packedWords(), logical.as<std::int8_t>(), result.rows_,
                                                    result.rowLength_, result.wordsPerRow_);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

DecodedTile TernaryTensor::decodeRows(std::size_t firstRow, std::size_t rowCount) const {
    if (firstRow + rowCount > rows_) {
        throw std::out_of_range("CUDA TernaryTensor::decodeRows: row range is outside the tensor");
    }
    DecodedTile result(rowCount, rowLength_);
    const std::size_t count = rowCount * rowLength_;
    if (count != 0) {
        decodeRowsKernel<<<gridFor(count), kBlockSize>>>(result.data(), packedWords(), scales(), firstRow, rowCount,
                                                        rowLength_, wordsPerRow_, groupsPerRow_, groupSize_);
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

blackforge::blackbit::TernaryTensor TernaryTensor::download() const {
    blackforge::blackbit::TernaryTensor result(shape_, groupSize_);
    BLACKFORGE_CUDA_CHECK(
        cudaMemcpy(result.packedWords().data(), packed_.data(), packed_.bytes(), cudaMemcpyDeviceToHost));
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(result.scales().data(), scales_.data(), scales_.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

}  // namespace blackforge::blackbit::cuda
