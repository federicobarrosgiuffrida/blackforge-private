#include "blackforge/blackbit/cuda_ternary_linear.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/device_shared.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;
constexpr std::size_t kCublasWorkspaceBytes = 4 * 1024 * 1024;

std::size_t rowsOf(const Tensor& tensor) {
    if (tensor.rank() < 2) {
        throw std::invalid_argument("CUDA TernaryLinear requires rank >= 2");
    }
    return tensor.elementCount() / tensor.shape().back();
}

void requireFeatures(const Tensor& tensor, std::size_t features, const std::string& name, const char* role) {
    if (tensor.rank() < 2 || tensor.shape().back() != features) {
        throw std::invalid_argument("CUDA TernaryLinear '" + name + "': invalid " + role + " shape");
    }
}

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__global__ void floatToBf16Kernel(__nv_bfloat16* output, const float* input, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2bfloat16(input[index]);
    }
}

__global__ void decodeBf16Kernel(__nv_bfloat16* output, const std::uint32_t* packed, const float* scales,
                                 std::size_t firstRow, std::size_t rowCount, std::size_t rowLength,
                                 std::size_t wordsPerRow, std::size_t groupsPerRow, std::size_t groupSize) {
    // La riga arriva da blockIdx.y e la colonna da blockIdx.x, invece di
    // essere ricavate da index/rowLength e index%rowLength: erano due
    // divisioni intere a 64 bit per ogni elemento decodificato, e la GPU
    // non ha una divisione hardware. Con un tile di 128 righe la griglia
    // 2D e' ampiamente entro il limite di 65535 blocchi in y.
    //
    // Le divisioni rimaste sono per costanti di compilazione (20 e 5,
    // che nvcc riduce a moltiplicazione e shift) e per groupSize, che a
    // 32 bit costa una frazione della versione a 64. Il valore prodotto
    // e' identico: cambia solo il modo di calcolare gli indici.
    const unsigned int localRow = blockIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= rowCount || col >= rowLength) {
        return;
    }
    const std::size_t row = firstRow + localRow;
    const unsigned int wordInRow = col / static_cast<unsigned int>(kTritsPerWord);
    const unsigned int inWord = col - wordInRow * static_cast<unsigned int>(kTritsPerWord);
    const std::uint32_t word = packed[row * wordsPerRow + wordInRow];
    const unsigned int byteIndex = inWord / static_cast<unsigned int>(kTritsPerByte);
    const unsigned int slot = inWord - byteIndex * static_cast<unsigned int>(kTritsPerByte);
    const int trit = decodeTritAt(wordByte(word, static_cast<int>(byteIndex)), static_cast<int>(slot));
    const unsigned int group = col / static_cast<unsigned int>(groupSize);
    output[static_cast<std::size_t>(localRow) * rowLength + col] =
        __float2bfloat16(static_cast<float>(trit) * scales[row * groupsPerRow + group]);
}

cublasLtHandle_t sharedHandle() {
    static std::unordered_map<int, cublasLtHandle_t> handles;
    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    auto found = handles.find(device);
    if (found != handles.end()) {
        return found->second;
    }
    cublasLtHandle_t handle = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtCreate(&handle));
    handles.emplace(device, handle);
    return handle;
}

Buffer& sharedWorkspace() {
    static Buffer workspace(kCublasWorkspaceBytes, MemoryArena::CublasWorkspace);
    return workspace;
}

struct LtDescriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    ~LtDescriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

void gemm(cublasOperation_t transA, cublasOperation_t transB, const __nv_bfloat16* a, std::size_t aRows,
          std::size_t aCols, std::size_t aLd, const __nv_bfloat16* b, std::size_t bRows, std::size_t bCols,
          std::size_t bLd, float* c, std::size_t cRows, std::size_t cCols, std::size_t cLd, float beta) {
    LtDescriptors descriptors;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                           &transA, sizeof(transA)));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                           &transB, sizeof(transB)));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descriptors.a, CUDA_R_16BF, aRows, aCols, aLd));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descriptors.b, CUDA_R_16BF, bRows, bCols, bLd));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&descriptors.c, CUDA_R_32F, cRows, cCols, cLd));
    cublasLtOrder_t rowMajor = CUBLASLT_ORDER_ROW;
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptors.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptors.b, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(descriptors.c, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&descriptors.preference));
    const std::size_t workspaceBytes = sharedWorkspace().bytes();
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        descriptors.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(sharedHandle(), descriptors.operation, descriptors.a,
                                                            descriptors.b, descriptors.c, descriptors.c,
                                                            descriptors.preference, 1, &heuristic, &returned));
    if (returned == 0) {
        throw std::runtime_error("CUDA TernaryLinear: cuBLASLt found no BF16 algorithm for the tile shape");
    }
    const float alpha = 1.0F;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmul(sharedHandle(), descriptors.operation, &alpha, a, descriptors.a, b,
                                            descriptors.b, &beta, c, descriptors.c, c, descriptors.c,
                                            &heuristic.algo, sharedWorkspace().data(), heuristic.workspaceSize, nullptr));
}

// Cronometro a eventi che NON blocca l'host.
//
// La versione precedente faceva cudaEventSynchronize dopo OGNI tile di
// decode e ogni GEMM. Il profilo Nsight contava 111 188 sincronizzazioni
// per step: l'host non poteva mai correre avanti a accodare il kernel
// successivo, e il tempo di parete restava ~2,4 s sopra la somma dei
// kernel. Qui le coppie di eventi vengono solo registrate; i tempi si
// leggono una volta sola alla fine, con una singola sincronizzazione.
// La metrica riportata resta la stessa, misurata dagli stessi eventi.
class DeferredTimer {
public:
    void reset() {
        pending_.clear();
        used_ = 0;
    }

    // Ogni marca occupa uno slot suo: gli intervalli possono annidarsi
    // (il cronometro totale racchiude quelli per tile) senza che una
    // start successiva sovrascriva l'evento di una ancora aperta.
    [[nodiscard]] std::size_t start() {
        const std::size_t index = mark();
        return index;
    }

    void stop(std::size_t startIndex, double* accumulator) {
        const std::size_t endIndex = mark();
        pending_.push_back(Interval{startIndex, endIndex, accumulator});
    }

    void resolve() {
        if (pending_.empty()) return;
        BLACKFORGE_CUDA_CHECK(cudaEventSynchronize(events_[used_ - 1]));
        for (const Interval& interval : pending_) {
            float milliseconds = 0.0F;
            BLACKFORGE_CUDA_CHECK(
                cudaEventElapsedTime(&milliseconds, events_[interval.start], events_[interval.end]));
            *interval.accumulator += milliseconds;
        }
        reset();
    }

private:
    struct Interval {
        std::size_t start;
        std::size_t end;
        double* accumulator;
    };

    std::size_t mark() {
        while (events_.size() <= used_) {
            cudaEvent_t event = nullptr;
            BLACKFORGE_CUDA_CHECK(cudaEventCreate(&event));
            events_.push_back(event);
        }
        const std::size_t index = used_++;
        BLACKFORGE_CUDA_CHECK(cudaEventRecord(events_[index]));
        return index;
    }

    std::vector<cudaEvent_t> events_;
    std::vector<Interval> pending_;
    std::size_t used_ = 0;
};

// Gli eventi sono riusati fra le chiamate: crearli e distruggerli a ogni
// forward costava 15 808 cudaEventCreate per step.
DeferredTimer& deferredTimer() {
    static thread_local DeferredTimer timer;
    return timer;
}

}  // namespace

GradientLifetimeStats& gradientLifetimeStats() {
    static GradientLifetimeStats stats;
    return stats;
}

void resetGradientLifetimeStats() { gradientLifetimeStats() = GradientLifetimeStats{}; }

TernaryLinear::TernaryLinear(const blackforge::blackbit::TernaryLinear& cpuReference)
    : name_(cpuReference.name()),
      inFeatures_(cpuReference.inFeatures()),
      outFeatures_(cpuReference.outFeatures()),
      tileRows_(cpuReference.tileRows()),
      weight_(cpuReference.weight()) {}

TernaryLinear::TernaryLinear(std::string name, std::size_t inFeatures, std::size_t outFeatures,
                             blackforge::blackbit::TernaryTensor hostWeight, std::size_t tileRows)
    : name_(std::move(name)),
      inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      tileRows_(std::min(tileRows, outFeatures)),
      weight_(hostWeight) {
    if (hostWeight.rows() != outFeatures_ || hostWeight.rowLength() != inFeatures_ || tileRows == 0) {
        throw std::invalid_argument("CUDA TernaryLinear: host weight or tile size does not match layer dimensions");
    }
}

Tensor TernaryLinear::forward(const Tensor& input) const {
    requireFeatures(input, inFeatures_, name_, "input");
    const std::size_t rows = rowsOf(input);
    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = outFeatures_;
    Tensor output(std::move(outputShape), MemoryArena::Activations);
    Buffer inputBf16(input.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer tile(tileRows_ * inFeatures_ * sizeof(__nv_bfloat16), MemoryArena::DequantizationTiles);

    metrics_ = {};
    metrics_.dequantWorkspaceBytes = tile.bytes();
    metrics_.cublasWorkspaceBytes = sharedWorkspace().bytes();

    // 'tile' viene riusato a ogni iterazione, ma decode e GEMM stanno
    // sullo stesso stream: l'ordine dello stream garantisce gia' che il
    // decode del tile successivo non parta prima che il GEMM precedente
    // abbia finito di leggerlo. La sincronizzazione dell'host serviva
    // solo a leggere il cronometro, non alla correttezza.
    DeferredTimer& timer = deferredTimer();
    timer.reset();
    const std::size_t totalMark = timer.start();
    floatToBf16Kernel<<<gridFor(input.elementCount()), kBlockSize>>>(inputBf16.as<__nv_bfloat16>(), input.data(),
                                                                    input.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());

    for (std::size_t first = 0; first < outFeatures_; first += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - first);
        const std::size_t decodeMark = timer.start();
        decodeBf16Kernel<<<dim3(gridFor(inFeatures_), static_cast<unsigned int>(count)), kBlockSize>>>(
            tile.as<__nv_bfloat16>(), weight_.packedWords(), weight_.scales(), first, count, inFeatures_,
            weight_.wordsPerRow(), weight_.groupsPerRow(), weight_.groupSize());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        timer.stop(decodeMark, &metrics_.decodeMs);

        const std::size_t gemmMark = timer.start();
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, inputBf16.as<__nv_bfloat16>(), rows, inFeatures_, inFeatures_,
             tile.as<__nv_bfloat16>(), count, inFeatures_, inFeatures_, output.data() + first, rows, count,
             outFeatures_, 0.0F);
        timer.stop(gemmMark, &metrics_.gemmMs);
        metrics_.decodedBytesWritten += count * inFeatures_ * sizeof(__nv_bfloat16);
        metrics_.packedBytesRead += count * weight_.wordsPerRow() * sizeof(std::uint32_t) +
                                    count * weight_.groupsPerRow() * sizeof(float);
    }
    timer.stop(totalMark, &metrics_.totalMs);
    timer.resolve();
    return output;
}

Tensor TernaryLinear::forwardRows(const Tensor& input, std::size_t firstRow, std::size_t rowCount) const {
    requireFeatures(input, inFeatures_, name_, "input");
    if (rowCount == 0 || firstRow + rowCount > outFeatures_) {
        throw std::invalid_argument("CUDA TernaryLinear '" + name_ + "': invalid forward row range");
    }
    const std::size_t rows = rowsOf(input);
    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = rowCount;
    Tensor output(std::move(outputShape), MemoryArena::Temporary);
    Buffer inputBf16(input.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    const std::size_t localTileRows = std::min(tileRows_, rowCount);
    Buffer tile(localTileRows * inFeatures_ * sizeof(__nv_bfloat16), MemoryArena::DequantizationTiles);
    floatToBf16Kernel<<<gridFor(input.elementCount()), kBlockSize>>>(
        inputBf16.as<__nv_bfloat16>(), input.data(), input.elementCount());
    for (std::size_t localFirst = 0; localFirst < rowCount; localFirst += localTileRows) {
        const std::size_t count = std::min(localTileRows, rowCount - localFirst);
        decodeBf16Kernel<<<dim3(gridFor(inFeatures_), static_cast<unsigned int>(count)), kBlockSize>>>(
            tile.as<__nv_bfloat16>(), weight_.packedWords(), weight_.scales(), firstRow + localFirst, count,
            inFeatures_, weight_.wordsPerRow(), weight_.groupsPerRow(), weight_.groupSize());
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, inputBf16.as<__nv_bfloat16>(), rows, inFeatures_, inFeatures_,
             tile.as<__nv_bfloat16>(), count, inFeatures_, inFeatures_, output.data() + localFirst, rows,
             count, rowCount, 0.0F);
    }
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor TernaryLinear::backwardInputRows(const Tensor& gradOutput, std::size_t firstRow) const {
    if (gradOutput.rank() < 2 || gradOutput.shape().back() == 0 ||
        firstRow + gradOutput.shape().back() > outFeatures_) {
        throw std::invalid_argument("CUDA TernaryLinear '" + name_ + "': invalid backward input row range");
    }
    const std::size_t rows = rowsOf(gradOutput);
    const std::size_t rowCount = gradOutput.shape().back();
    std::vector<std::size_t> resultShape = gradOutput.shape();
    resultShape.back() = inFeatures_;
    Tensor result = Tensor::zeros(std::move(resultShape), MemoryArena::Activations);
    Buffer gradBf16(gradOutput.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    const std::size_t localTileRows = std::min(tileRows_, rowCount);
    Buffer tile(localTileRows * inFeatures_ * sizeof(__nv_bfloat16), MemoryArena::DequantizationTiles);
    floatToBf16Kernel<<<gridFor(gradOutput.elementCount()), kBlockSize>>>(
        gradBf16.as<__nv_bfloat16>(), gradOutput.data(), gradOutput.elementCount());
    for (std::size_t localFirst = 0; localFirst < rowCount; localFirst += localTileRows) {
        const std::size_t count = std::min(localTileRows, rowCount - localFirst);
        decodeBf16Kernel<<<dim3(gridFor(inFeatures_), static_cast<unsigned int>(count)), kBlockSize>>>(
            tile.as<__nv_bfloat16>(), weight_.packedWords(), weight_.scales(), firstRow + localFirst, count,
            inFeatures_, weight_.wordsPerRow(), weight_.groupsPerRow(), weight_.groupSize());
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, gradBf16.as<__nv_bfloat16>() + localFirst, rows, count, rowCount,
             tile.as<__nv_bfloat16>(), count, inFeatures_, inFeatures_, result.data(), rows, inFeatures_,
             inFeatures_, localFirst == 0 ? 0.0F : 1.0F);
    }
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor TernaryLinear::weightGradientRows(const Tensor& input, const Tensor& gradOutput,
                                         std::size_t firstRow) const {
    requireFeatures(input, inFeatures_, name_, "input");
    if (gradOutput.rank() < 2 || rowsOf(input) != rowsOf(gradOutput) ||
        gradOutput.shape().back() == 0 || firstRow + gradOutput.shape().back() > outFeatures_) {
        throw std::invalid_argument("CUDA TernaryLinear '" + name_ + "': invalid weight-gradient row range");
    }
    const std::size_t rows = rowsOf(input);
    const std::size_t rowCount = gradOutput.shape().back();
    Tensor result({rowCount, inFeatures_}, MemoryArena::GradientTiles);
    Buffer inputBf16(input.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer gradBf16(gradOutput.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    floatToBf16Kernel<<<gridFor(input.elementCount()), kBlockSize>>>(
        inputBf16.as<__nv_bfloat16>(), input.data(), input.elementCount());
    floatToBf16Kernel<<<gridFor(gradOutput.elementCount()), kBlockSize>>>(
        gradBf16.as<__nv_bfloat16>(), gradOutput.data(), gradOutput.elementCount());
    gemm(CUBLAS_OP_T, CUBLAS_OP_N, gradBf16.as<__nv_bfloat16>(), rows, rowCount, rowCount,
         inputBf16.as<__nv_bfloat16>(), rows, inFeatures_, inFeatures_, result.data(), rowCount,
         inFeatures_, inFeatures_, 0.0F);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor TernaryLinear::backward(const Tensor& input, const Tensor& gradOutput, GradientSink* sink) const {
    requireFeatures(input, inFeatures_, name_, "input");
    requireFeatures(gradOutput, outFeatures_, name_, "output gradient");
    const std::size_t rows = rowsOf(input);
    if (rowsOf(gradOutput) != rows) {
        throw std::invalid_argument("CUDA TernaryLinear: input and output gradient row counts differ");
    }

    Tensor gradInput = Tensor::zeros(input.shape(), MemoryArena::Activations);
    Buffer inputBf16(input.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer gradOutputBf16(gradOutput.elementCount() * sizeof(__nv_bfloat16), MemoryArena::Temporary);
    Buffer tile(tileRows_ * inFeatures_ * sizeof(__nv_bfloat16), MemoryArena::DequantizationTiles);
    Buffer gradTile(tileRows_ * inFeatures_ * sizeof(float), MemoryArena::GradientTiles);
    floatToBf16Kernel<<<gridFor(input.elementCount()), kBlockSize>>>(inputBf16.as<__nv_bfloat16>(), input.data(),
                                                                    input.elementCount());
    floatToBf16Kernel<<<gridFor(gradOutput.elementCount()), kBlockSize>>>(
        gradOutputBf16.as<__nv_bfloat16>(), gradOutput.data(), gradOutput.elementCount());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());

    GradientLifetimeStats& lifetime = gradientLifetimeStats();
    for (std::size_t first = 0; first < outFeatures_; first += tileRows_) {
        const std::size_t count = std::min(tileRows_, outFeatures_ - first);
        const std::size_t gradientBytes = count * inFeatures_ * sizeof(float);
        decodeBf16Kernel<<<dim3(gridFor(inFeatures_), static_cast<unsigned int>(count)), kBlockSize>>>(
            tile.as<__nv_bfloat16>(), weight_.packedWords(), weight_.scales(), first, count, inFeatures_,
            weight_.wordsPerRow(), weight_.groupsPerRow(), weight_.groupSize());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());

        // dX += dY_tile @ W_tile. gradOutput is strided by outFeatures.
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, gradOutputBf16.as<__nv_bfloat16>() + first, rows, count, outFeatures_,
             tile.as<__nv_bfloat16>(), count, inFeatures_, inFeatures_, gradInput.data(), rows, inFeatures_,
             inFeatures_, first == 0 ? 0.0F : 1.0F);
        // dW_tile = dY_tile^T @ X, produced and consumed before the next tile.
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, gradOutputBf16.as<__nv_bfloat16>() + first, rows, count, outFeatures_,
             inputBf16.as<__nv_bfloat16>(), rows, inFeatures_, inFeatures_, gradTile.as<float>(), count,
             inFeatures_, inFeatures_, 0.0F);

        lifetime.liveBytes = gradientBytes;
        lifetime.peakLiveBytes = std::max(lifetime.peakLiveBytes, gradientBytes);
        lifetime.cumulativeBytes += gradientBytes;
        ++lifetime.blocksProduced;
        if (sink != nullptr) {
            sink->consumeWeightGradientBlock(parameterId(), first, count, gradTile.as<float>());
        }
        lifetime.liveBytes = 0;
        ++lifetime.blocksReleased;
    }
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());
    return gradInput;
}

}  // namespace blackforge::blackbit::cuda
