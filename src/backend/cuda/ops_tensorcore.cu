#include <cuda_bf16.h>
#include <cublasLt.h>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "blackforge/backend/cuda/autodiff.hpp"
#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/backend/cuda/device_pool.hpp"
#include "blackforge/backend/cuda/ops.hpp"

namespace blackforge::backend::cuda {

namespace {

constexpr int kBlockSize = 256;
int gridSizeFor(std::size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__global__ void floatToBf16Kernel(__nv_bfloat16* out, const float* in, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(in[i]);
    }
}

// Buffer BF16 grezzo su device (RAII): DeviceTensor memorizza sempre
// float32 per contratto (vedi device_tensor.hpp), quindi lo scratch
// BF16 usato solo internamente da questo file per alimentare Tensor
// Core non puo' essere un DeviceTensor — e' un dettaglio implementativo
// che non deve mai attraversare l'API pubblica (che resta sempre
// float32 in ingresso/uscita, vedi matmulBf16 sotto).
class DeviceBf16Buffer {
public:
    explicit DeviceBf16Buffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            // Stesso pool per-device di DeviceTensor (vedi
            // device_pool.hpp): questo buffer viene ri-allocato ad ogni
            // singola chiamata a matmulBf16/linearBf16, con dimensioni
            // che si ripetono identiche ad ogni step di un training
            // loop — lo stesso ragionamento che ha motivato il pool per
            // DeviceTensor si applica qui.
            ptr_ = static_cast<__nv_bfloat16*>(devicePoolAcquire(count_ * sizeof(__nv_bfloat16)));
        }
    }
    ~DeviceBf16Buffer() {
        if (ptr_ != nullptr) {
            devicePoolRelease(ptr_, count_ * sizeof(__nv_bfloat16));
        }
    }
    DeviceBf16Buffer(const DeviceBf16Buffer&) = delete;
    DeviceBf16Buffer& operator=(const DeviceBf16Buffer&) = delete;

    DeviceBf16Buffer(DeviceBf16Buffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBf16Buffer& operator=(DeviceBf16Buffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                devicePoolRelease(ptr_, count_ * sizeof(__nv_bfloat16));
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    [[nodiscard]] __nv_bfloat16* data() { return ptr_; }
    [[nodiscard]] const __nv_bfloat16* data() const { return ptr_; }

private:
    __nv_bfloat16* ptr_ = nullptr;
    std::size_t count_;
};

DeviceBf16Buffer toBf16(const DeviceTensor& t) {
    DeviceBf16Buffer buffer(t.elementCount());
    if (t.elementCount() > 0) {
        floatToBf16Kernel<<<gridSizeFor(t.elementCount()), kBlockSize>>>(buffer.data(), t.data(), t.elementCount());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    }
    return buffer;
}

// Cache dei pesi convertiti in BF16, per device (stessa esigenza di
// gemmPlanRegistry() sotto: repliche multi-GPU hanno buffer di pesi
// indipendenti, un puntatore da solo non e' univoco tra device diversi).
// Chiave: il puntatore device del peso FP32; valore: la generazione in
// cui e' stata calcolata la conversione, piu' il buffer BF16 stesso. Un
// hit e' valido solo se la generazione salvata coincide con quella
// corrente (vedi invalidateBf16WeightCache() sotto) — necessario perche'
// il pool di memoria (device_pool.hpp) puo' restituire lo stesso
// puntatore per un buffer riallocato con valori diversi.
struct CachedBf16Weight {
    std::size_t generation;
    DeviceBf16Buffer buffer;
};

std::size_t& bf16WeightGeneration() {
    static std::size_t generation = 0;
    return generation;
}

std::unordered_map<int, std::unordered_map<const float*, CachedBf16Weight>>& bf16WeightCacheRegistry() {
    static std::unordered_map<int, std::unordered_map<const float*, CachedBf16Weight>> registry;
    return registry;
}

// Come toBf16(), ma pensata SOLO per l'operando "peso" di matmulBf16()/
// matmulBf16Backward() (mai per l'attivazione, che cambia davvero ad
// ogni chiamata): riusa la conversione BF16 dell'ultima chiamata con lo
// stesso puntatore, se calcolata nella generazione corrente.
const DeviceBf16Buffer& toBf16Weight(const DeviceTensor& t) {
    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    std::unordered_map<const float*, CachedBf16Weight>& perDevice = bf16WeightCacheRegistry()[device];
    std::size_t currentGeneration = bf16WeightGeneration();

    auto it = perDevice.find(t.data());
    if (it != perDevice.end() && it->second.generation == currentGeneration) {
        return it->second.buffer;
    }

    CachedBf16Weight entry{currentGeneration, toBf16(t)};
    auto [insertedIt, inserted] = perDevice.insert_or_assign(t.data(), std::move(entry));
    return insertedIt->second.buffer;
}

// Un handle cuBLASLt per DISPOSITIVO CUDA, stessa scelta e stesso
// motivo di sharedHandle() in ops_gemm.cu per il cuBLAS classico
// (essenziale per il training multi-GPU: un handle creato mentre e'
// attivo un device non e' valido su un device diverso).
cublasLtHandle_t sharedLtHandle() {
    static std::unordered_map<int, cublasLtHandle_t> handles;
    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    auto it = handles.find(device);
    if (it != handles.end()) {
        return it->second;
    }
    cublasLtHandle_t h;
    BLACKFORGE_CUBLAS_CHECK(cublasLtCreate(&h));
    handles.emplace(device, h);
    return h;
}

// Un "piano" di esecuzione GEMM completamente risolto: descrittore
// dell'operazione, i tre layout (A/B/C), l'algoritmo scelto
// dall'euristica cuBLASLt e il workspace device gia' allocato — tutto
// cio' che serve per lanciare cublasLtMatmul() senza ricalcolare o
// riallocare nulla.
struct GemmPlan {
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aLayout = nullptr;
    cublasLtMatrixLayout_t bLayout = nullptr;
    cublasLtMatrixLayout_t cLayout = nullptr;
    cublasLtMatmulAlgo_t algo{};
    void* workspace = nullptr;
    std::size_t workspaceBytes = 0;
};

// Chiave di cache: le SEI dimensioni (cosi' come memorizzate, prima di
// un'eventuale trasposizione logica) piu' le due direzioni di
// trasposizione determinano univocamente i tre layout e la scelta
// dell'algoritmo — non e' sufficiente un sottoinsieme (M,K,N), perche'
// aRows/bRows/bCols dipendono da COME l'operando e' memorizzato, non
// solo dalla forma logica del prodotto (vedi matmulBf16Backward: la
// stessa tripletta (M,K,N) logica puo' corrispondere a operandi
// memorizzati con dimensioni diverse a seconda di quale side viene
// trasposto).
using GemmPlanKey =
    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, int, int>;

// Un piano per DEVICE CUDA e per forma: le forme di un training loop
// (stesso modello, stesso batch_size) si ripetono IDENTICHE ad ogni
// step, quindi calcolare/allocare tutto questo una sola volta invece di
// farlo ad ogni singola chiamata a matmulBf16/matmulBf16Backward (fino
// a ~150 volte per step in un modello con 8 layer attention+feedforward)
// e' il guadagno principale di questa cache — la ricerca euristica
// dell'algoritmo (cublasLtMatmulAlgoGetHeuristic) e soprattutto il
// cudaMalloc/cudaFree diretto del workspace (che PRIMA bypassava
// interamente il pool di device_pool.hpp) erano il costo dominante per
// chiamata, misurato essere la causa principale del gap di prestazioni
// residuo rispetto a un training loop PyTorch equivalente. Mai
// invalidata/liberata esplicitamente (stessa scelta degli handle
// cuBLAS/cuBLASLt condivisi): le forme di un modello non cambiano mai
// dopo la costruzione.
std::unordered_map<int, std::map<GemmPlanKey, GemmPlan>>& gemmPlanRegistry() {
    static std::unordered_map<int, std::map<GemmPlanKey, GemmPlan>> registry;
    return registry;
}

GemmPlan& getOrCreateGemmPlan(cublasOperation_t transA, cublasOperation_t transB, std::size_t aRows,
                               std::size_t aCols, std::size_t bRows, std::size_t bCols, std::size_t resultRows,
                               std::size_t resultCols) {
    int device = 0;
    BLACKFORGE_CUDA_CHECK(cudaGetDevice(&device));
    GemmPlanKey key{aRows, aCols, bRows, bCols, resultRows, resultCols, static_cast<int>(transA),
                     static_cast<int>(transB)};

    std::map<GemmPlanKey, GemmPlan>& perDevice = gemmPlanRegistry()[device];
    auto it = perDevice.find(key);
    if (it != perDevice.end()) {
        return it->second;
    }

    cublasLtHandle_t handle = sharedLtHandle();
    GemmPlan plan;

    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.aLayout, CUDA_R_16BF, static_cast<std::uint64_t>(aRows),
                                                        static_cast<std::uint64_t>(aCols),
                                                        static_cast<std::int64_t>(aCols)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(plan.aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.bLayout, CUDA_R_16BF, static_cast<std::uint64_t>(bRows),
                                                        static_cast<std::uint64_t>(bCols),
                                                        static_cast<std::int64_t>(bCols)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(plan.bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    BLACKFORGE_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.cLayout, CUDA_R_32F,
                                                        static_cast<std::uint64_t>(resultRows),
                                                        static_cast<std::uint64_t>(resultCols),
                                                        static_cast<std::int64_t>(resultCols)));
    BLACKFORGE_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(plan.cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    cublasLtMatmulPreference_t preference = nullptr;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    std::size_t maxWorkspaceBytes = 4 * 1024 * 1024;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspaceBytes, sizeof(maxWorkspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasStatus_t heurStatus =
        cublasLtMatmulAlgoGetHeuristic(handle, plan.opDesc, plan.aLayout, plan.bLayout, plan.cLayout, plan.cLayout,
                                        preference, 1, &heuristicResult, &returnedResults);
    cublasLtMatmulPreferenceDestroy(preference);  // non serve piu' dopo la ricerca dell'algoritmo

    if (heurStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        cublasLtMatrixLayoutDestroy(plan.aLayout);
        cublasLtMatrixLayoutDestroy(plan.bLayout);
        cublasLtMatrixLayoutDestroy(plan.cLayout);
        cublasLtMatmulDescDestroy(plan.opDesc);
        throw std::runtime_error("bf16Gemm: nessun algoritmo Tensor Core disponibile per questa forma/architettura");
    }

    plan.algo = heuristicResult.algo;
    plan.workspaceBytes = heuristicResult.workspaceSize;
    if (plan.workspaceBytes > 0) {
        // Workspace persistente per questa forma (mai liberato
        // esplicitamente, come il resto del piano): un cudaMalloc unico
        // invece di uno per chiamata, e passa dal pool di
        // device_pool.hpp invece di un cudaMalloc/cudaFree diretto come
        // in precedenza.
        plan.workspace = devicePoolAcquire(plan.workspaceBytes);
    }

    auto [insertedIt, inserted] = perDevice.emplace(key, plan);
    return insertedIt->second;
}

// Esegue C[resultRows, resultCols] = op(A) @ op(B) in row-major, con A
// e B gia' in BF16 (Tensor Core) e accumulo/uscita in FP32 (lo schema
// di "mixed precision" standard: attivazioni/pesi in BF16 per il
// calcolo, ma niente overflow/perdita di range nell'accumulo, a
// differenza di un accumulo BF16 puro). 'aRows'/'aCols' e
// 'bRows'/'bCols' sono le dimensioni di A/B COSI' COME MEMORIZZATE
// (prima di un'eventuale trasposizione logica via transA/transB): la
// stessa convenzione di cuBLAS/cuBLASLt classici. A differenza del
// trucco riga-maggiore/colonna-maggiore usato da matmul() (ops_gemm.cu,
// cuBLAS classico), cuBLASLt supporta nativamente CUBLASLT_ORDER_ROW,
// quindi qui non serve scambiare operandi/dimensioni.
//
// Il piano (descrittori, layout, algoritmo, workspace) viene calcolato
// una sola volta per forma e riusato da ogni chiamata successiva con la
// stessa forma (vedi getOrCreateGemmPlan/GemmPlan sopra): le forme di
// un training loop si ripetono identiche ad ogni step.
void bf16Gemm(cublasOperation_t transA, cublasOperation_t transB, const __nv_bfloat16* a, std::size_t aRows,
              std::size_t aCols, const __nv_bfloat16* b, std::size_t bRows, std::size_t bCols, float* c,
              std::size_t resultRows, std::size_t resultCols) {
    cublasLtHandle_t handle = sharedLtHandle();
    GemmPlan& plan = getOrCreateGemmPlan(transA, transB, aRows, aCols, bRows, bCols, resultRows, resultCols);

    float alpha = 1.0F;
    float beta = 0.0F;
    BLACKFORGE_CUBLAS_CHECK(cublasLtMatmul(handle, plan.opDesc, &alpha, a, plan.aLayout, b, plan.bLayout, &beta, c,
                                            plan.cLayout, c, plan.cLayout, &plan.algo, plan.workspace,
                                            plan.workspaceBytes, /*stream=*/nullptr));
}

}  // namespace

DeviceTensor matmulBf16(const DeviceTensor& a, const DeviceTensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmulBf16: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceBf16Buffer aBf16 = toBf16(a);
    DeviceBf16Buffer bBf16 = toBf16(b);
    DeviceTensor result({m, n});

    if (m > 0 && k > 0 && n > 0) {
        bf16Gemm(CUBLAS_OP_N, CUBLAS_OP_N, aBf16.data(), m, k, bBf16.data(), k, n, result.data(), m, n);
    } else {
        BLACKFORGE_CUDA_CHECK(cudaMemset(result.data(), 0, result.elementCount() * sizeof(float)));
    }

    return result;
}

DeviceTensor matmulBf16CachedWeight(const DeviceTensor& a, const DeviceTensor& b) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmulBf16CachedWeight: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceBf16Buffer aBf16 = toBf16(a);
    const DeviceBf16Buffer& bBf16 = toBf16Weight(b);
    DeviceTensor result({m, n});

    if (m > 0 && k > 0 && n > 0) {
        bf16Gemm(CUBLAS_OP_N, CUBLAS_OP_N, aBf16.data(), m, k, bBf16.data(), k, n, result.data(), m, n);
    } else {
        BLACKFORGE_CUDA_CHECK(cudaMemset(result.data(), 0, result.elementCount() * sizeof(float)));
    }

    return result;
}

DeviceTensor linearBf16(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias) {
    if (input.rank() == 2) {
        return addBias(matmulBf16(input, weight), bias);
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("linearBf16: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t inFeatures = input.shape().back();
    std::size_t rows = input.elementCount() / inFeatures;
    DeviceTensor flatOutput = addBias(matmulBf16(input.clone().reshaped({rows, inFeatures}), weight), bias);

    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = weight.dim(1);
    return std::move(flatOutput).reshaped(std::move(outputShape));
}

// Come linearBf16(), ma usa matmulBf16CachedWeight() al posto di
// matmulBf16(): stessa precondizione/contratto di matmulBf16CachedWeight()
// (vedi ops.hpp), pensata solo per l'uso interno di
// feedForwardForwardCached().
DeviceTensor linearBf16CachedWeight(const DeviceTensor& input, const DeviceTensor& weight, const DeviceTensor& bias) {
    if (input.rank() == 2) {
        return addBias(matmulBf16CachedWeight(input, weight), bias);
    }
    if (input.rank() < 2) {
        throw std::invalid_argument("linearBf16CachedWeight: richiede un tensore a rango >= 2 sul device");
    }

    std::size_t inFeatures = input.shape().back();
    std::size_t rows = input.elementCount() / inFeatures;
    DeviceTensor flatOutput =
        addBias(matmulBf16CachedWeight(input.clone().reshaped({rows, inFeatures}), weight), bias);

    std::vector<std::size_t> outputShape = input.shape();
    outputShape.back() = weight.dim(1);
    return std::move(flatOutput).reshaped(std::move(outputShape));
}

MatmulGrad matmulBf16Backward(const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(0) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulBf16Backward: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceBf16Buffer aBf16 = toBf16(a);
    DeviceBf16Buffer bBf16 = toBf16(b);
    DeviceBf16Buffer gradBf16 = toBf16(gradOutput);

    DeviceTensor dA({m, k});
    DeviceTensor dB({k, n});

    if (m > 0 && k > 0 && n > 0) {
        // dA[m,k] = gradOutput[m,n] @ B^T: op(gradOutput) senza
        // trasposizione, op(B) trasposta (B e' memorizzata [k,n]).
        bf16Gemm(CUBLAS_OP_N, CUBLAS_OP_T, gradBf16.data(), m, n, bBf16.data(), k, n, dA.data(), m, k);
        // dB[k,n] = A^T @ gradOutput: op(A) trasposta (A e' memorizzata
        // [m,k]), op(gradOutput) senza trasposizione.
        bf16Gemm(CUBLAS_OP_T, CUBLAS_OP_N, aBf16.data(), m, k, gradBf16.data(), m, n, dB.data(), k, n);
    } else {
        BLACKFORGE_CUDA_CHECK(cudaMemset(dA.data(), 0, dA.elementCount() * sizeof(float)));
        BLACKFORGE_CUDA_CHECK(cudaMemset(dB.data(), 0, dB.elementCount() * sizeof(float)));
    }

    return MatmulGrad{std::move(dA), std::move(dB)};
}

MatmulGrad matmulBf16BackwardCachedWeight(const DeviceTensor& a, const DeviceTensor& b,
                                           const DeviceTensor& gradOutput) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(0) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(1)) {
        throw std::invalid_argument("matmulBf16BackwardCachedWeight: forme incompatibili sul device");
    }
    std::size_t m = a.dim(0);
    std::size_t k = a.dim(1);
    std::size_t n = b.dim(1);

    DeviceBf16Buffer aBf16 = toBf16(a);
    const DeviceBf16Buffer& bBf16 = toBf16Weight(b);
    DeviceBf16Buffer gradBf16 = toBf16(gradOutput);

    DeviceTensor dA({m, k});
    DeviceTensor dB({k, n});

    if (m > 0 && k > 0 && n > 0) {
        bf16Gemm(CUBLAS_OP_N, CUBLAS_OP_T, gradBf16.data(), m, n, bBf16.data(), k, n, dA.data(), m, k);
        bf16Gemm(CUBLAS_OP_T, CUBLAS_OP_N, aBf16.data(), m, k, gradBf16.data(), m, n, dB.data(), k, n);
    } else {
        BLACKFORGE_CUDA_CHECK(cudaMemset(dA.data(), 0, dA.elementCount() * sizeof(float)));
        BLACKFORGE_CUDA_CHECK(cudaMemset(dB.data(), 0, dB.elementCount() * sizeof(float)));
    }

    return MatmulGrad{std::move(dA), std::move(dB)};
}

void invalidateBf16WeightCache() { ++bf16WeightGeneration(); }

}  // namespace blackforge::backend::cuda
