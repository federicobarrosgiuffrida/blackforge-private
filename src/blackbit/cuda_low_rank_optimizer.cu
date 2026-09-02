#include "blackforge/blackbit/cuda_low_rank_optimizer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"
#include "blackforge/blackbit/cuda_profile.hpp"
#include "blackforge/blackbit/stochastic_round.hpp"
#include "blackforge/blackbit/ternary_update.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr int kBlockSize = 256;
constexpr int kCudaWarpSizeOptimizer = 32;

struct DeviceStepStats {
    unsigned long long elements;
    unsigned long long flips;
    unsigned long long positiveFlips;
    unsigned long long negativeFlips;
    unsigned long long saturated;
    unsigned long long nanInf;
    unsigned long long gradientCount;
    unsigned long long optimizerCount;
    unsigned long long updateCount;
    double gradientSquared;
    double optimizerSquared;
    double updateSquared;
};

unsigned int gridFor(std::size_t count) {
    return static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
}

__device__ float projection(std::uint64_t seed, std::uint64_t epoch, std::size_t row, std::size_t component,
                            std::size_t rank) {
    const std::uint64_t mixed = counterRandom(seed ^ (epoch * 0x9E3779B97F4A7C15ULL),
                                               static_cast<std::uint64_t>(row) * rank + component);
    return ((mixed & 1ULL) != 0 ? 1.0F : -1.0F) / sqrtf(static_cast<float>(rank));
}

__global__ void gradientStatsKernel(const float* gradient, std::size_t count, DeviceStepStats* stats) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ double squared[kBlockSize];
    __shared__ unsigned int valid[kBlockSize];
    __shared__ unsigned int invalid[kBlockSize];
    double localSquared = 0.0;
    unsigned int localValid = 0;
    unsigned int localInvalid = 0;
    if (index < count) {
        const float value = gradient[index];
        if (isfinite(value)) {
            localSquared = static_cast<double>(value) * static_cast<double>(value);
            localValid = 1;
        } else {
            localInvalid = 1;
        }
    }
    squared[threadIdx.x] = localSquared;
    valid[threadIdx.x] = localValid;
    invalid[threadIdx.x] = localInvalid;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            squared[threadIdx.x] += squared[threadIdx.x + stride];
            valid[threadIdx.x] += valid[threadIdx.x + stride];
            invalid[threadIdx.x] += invalid[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats->gradientSquared, squared[0]);
        atomicAdd(&stats->gradientCount, static_cast<unsigned long long>(valid[0]));
        if (invalid[0] != 0) atomicAdd(&stats->nanInf, static_cast<unsigned long long>(invalid[0]));
    }
}

// Una componente per blocco: cosi' tutti i thread del blocco condividono
// lo stesso 'component' per costruzione, anche quando cols non e' un
// multiplo della dimensione del blocco.
//
// projection() dipende da (riga, componente) e non dalla colonna: nel
// ciclo originale ogni thread la ricalcolava per ogni riga, quindi i 256
// thread di un blocco valutavano 256 volte gli stessi hash. Qui il
// blocco li calcola una volta in memoria condivisa e li riusa.
//
// Le somme restano per riga crescente, con gli stessi raggruppamenti,
// quindi l'accumulatore e' bit-identico a prima.
// I blocchi con component == 0 leggono ogni elemento del blocco di
// gradiente esattamente una volta, quindi calcolano anche le statistiche
// del gradiente riusando il valore gia' in registro. Prima serviva un
// gradientStatsKernel separato che rileggeva l'intero blocco: una
// seconda passata completa sulla stessa memoria, per una metrica
// diagnostica. I conteggi sono interi e restano identici; per la somma
// dei quadrati cambia solo l'ordine di accumulazione, e alimenta solo il
// gradientRms riportato, non il clipping ne' altre decisioni.
__global__ void projectGradientKernel(float* accumulator, const float* gradient, std::size_t firstRow,
                                      std::size_t rowCount, std::size_t cols, std::size_t rank,
                                      std::uint64_t seed, std::uint64_t epoch, DeviceStepStats* stats) {
    const std::size_t component = blockIdx.y;
    if (component >= rank) return;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = col < cols;
    const bool collectStats = component == 0;
    __shared__ float sharedProjection[kBlockSize];
    float sum = active ? accumulator[component * cols + col] : 0.0F;
    double statsSquared = 0.0;
    unsigned int statsValid = 0;
    unsigned int statsInvalid = 0;
    for (std::size_t base = 0; base < rowCount; base += kBlockSize) {
        const std::size_t chunk = min(static_cast<std::size_t>(kBlockSize), rowCount - base);
        if (threadIdx.x < chunk) {
            sharedProjection[threadIdx.x] =
                projection(seed, epoch, firstRow + base + threadIdx.x, component, rank);
        }
        __syncthreads();
        if (active) {
            for (std::size_t k = 0; k < chunk; ++k) {
                const float value = gradient[(base + k) * cols + col];
                sum += sharedProjection[k] * value;
                if (collectStats) {
                    if (isfinite(value)) {
                        statsSquared += static_cast<double>(value) * static_cast<double>(value);
                        ++statsValid;
                    } else {
                        ++statsInvalid;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (active) accumulator[component * cols + col] = sum;
    if (!collectStats) return;

    constexpr unsigned int kFullMask = 0xFFFFFFFFU;
    constexpr int kWarpSize = 32;
    constexpr int kWarps = kBlockSize / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        statsSquared += __shfl_down_sync(kFullMask, statsSquared, offset);
        statsValid += __shfl_down_sync(kFullMask, statsValid, offset);
        statsInvalid += __shfl_down_sync(kFullMask, statsInvalid, offset);
    }
    __shared__ double sharedSquared[kWarps];
    __shared__ unsigned int sharedValid[kWarps];
    __shared__ unsigned int sharedInvalid[kWarps];
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    if (lane == 0) {
        sharedSquared[warp] = statsSquared;
        sharedValid[warp] = statsValid;
        sharedInvalid[warp] = statsInvalid;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    double squared = 0.0;
    unsigned long long valid = 0;
    unsigned long long invalid = 0;
    for (int w = 0; w < kWarps; ++w) {
        squared += sharedSquared[w];
        valid += sharedValid[w];
        invalid += sharedInvalid[w];
    }
    if (valid != 0) {
        atomicAdd(&stats->gradientSquared, squared);
        atomicAdd(&stats->gradientCount, valid);
    }
    if (invalid != 0) atomicAdd(&stats->nanInf, invalid);
}

__global__ void accumulateDenseKernel(float* accumulator, const float* gradient, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) accumulator[index] += gradient[index];
}

__global__ void squaredNormKernel(const float* values, std::size_t count, double* squared) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const double value = values[index];
        atomicAdd(squared, value * value);
    }
}

// Gradiente rispetto alla scala di un gruppo.
//
// Un peso ternario vale W_ij = trit_ij * s_g, quindi dW_ij/ds_g = trit_ij e
// dL/ds_g = somma sui pesi del gruppo di dL/dW_ij * trit_ij. Il blocco di
// gradiente arriva a pezzi di righe, percio' ogni lancio accumula solo il
// contributo delle righe che ha in mano; l'accumulatore viene azzerato in
// endStep dopo l'aggiornamento, come per la proiezione.
//
// Un blocco = un warp = una coppia (riga, gruppo). Il gruppo copre groupSize
// colonne consecutive e l'ultimo gruppo di una riga puo' essere parziale
// quando cols non e' multiplo di groupSize.
__global__ void scaleGradientKernel(float* scaleAccumulator, const float* block,
                                    const std::uint32_t* packed, std::size_t firstRow,
                                    std::size_t rowCount, std::size_t cols, std::size_t wordsPerRow,
                                    std::size_t groupsPerRow, std::size_t groupSize) {
    const std::size_t localRow = blockIdx.y;
    const std::size_t group = blockIdx.x;
    if (localRow >= rowCount || group >= groupsPerRow) return;
    const std::size_t row = firstRow + localRow;
    const std::size_t firstCol = group * groupSize;
    if (firstCol >= cols) return;
    const std::size_t lastCol = min(firstCol + groupSize, cols);

    float partial = 0.0F;
    for (std::size_t col = firstCol + threadIdx.x; col < lastCol; col += blockDim.x) {
        const std::size_t wordInRow = col / kTritsPerWord;
        const std::size_t inWord = col - wordInRow * kTritsPerWord;
        const std::uint32_t word = packed[row * wordsPerRow + wordInRow];
        const std::size_t byteIndex = inWord / kTritsPerByte;
        const std::size_t slot = inWord - byteIndex * kTritsPerByte;
        const int trit = decodeTritAt(wordByte(word, static_cast<int>(byteIndex)), static_cast<int>(slot));
        partial += block[localRow * cols + col] * static_cast<float>(trit);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&scaleAccumulator[row * groupsPerRow + group], partial);
    }
}

__global__ void denseAdamKernel(float* values, float* firstMoment, float* secondMoment, float* accumulator,
                                std::size_t count, float beta1, float beta2, float bias1, float bias2,
                                float epsilon, float learningRate, float weightDecay, float parameterScale,
                                DeviceStepStats* stats) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float gradient = accumulator[index];
    const float first = beta1 * firstMoment[index] + (1.0F - beta1) * gradient;
    const float second = beta2 * secondMoment[index] + (1.0F - beta2) * gradient * gradient;
    firstMoment[index] = first;
    secondMoment[index] = second;
    const float direction = (first / bias1) / (sqrtf(second / bias2) + epsilon);
    float parameter = values[index];
    const float update = -learningRate * parameterScale * direction;
    parameter += update;
    parameter -= learningRate * weightDecay * parameter;
    values[index] = parameter;
    accumulator[index] = 0.0F;
    if (!isfinite(parameter) || !isfinite(direction)) {
        atomicAdd(&stats->nanInf, 1ULL);
        return;
    }
    atomicAdd(&stats->optimizerSquared, static_cast<double>(direction) * static_cast<double>(direction));
    atomicAdd(&stats->optimizerCount, 1ULL);
    atomicAdd(&stats->updateSquared, static_cast<double>(update) * static_cast<double>(update));
    atomicAdd(&stats->updateCount, 1ULL);
}

__global__ void adamDirectionKernel(float* firstMoment, float* secondMoment, float* accumulator,
                                    std::size_t count, float beta1, float beta2, float bias1, float bias2,
                                    float epsilon, DeviceStepStats* stats) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float gradient = accumulator[index];
    const float first = beta1 * firstMoment[index] + (1.0F - beta1) * gradient;
    const float second = beta2 * secondMoment[index] + (1.0F - beta2) * gradient * gradient;
    firstMoment[index] = first;
    secondMoment[index] = second;
    const float direction = (first / bias1) / (sqrtf(second / bias2) + epsilon);
    accumulator[index] = direction;
    if (!isfinite(direction)) {
        atomicAdd(&stats->nanInf, 1ULL);
        return;
    }
    atomicAdd(&stats->optimizerSquared, static_cast<double>(direction) * static_cast<double>(direction));
    atomicAdd(&stats->optimizerCount, 1ULL);
}

// Numero di componenti caricate in memoria condivisa per volta. Fissa,
// non pari a 'rank', cosi' la memoria condivisa non dipende da un valore
// di runtime: kComponentTile * kTritsPerWord * 4 = 2560 byte per blocco,
// che non limita l'occupancy con nessun rango configurabile.
constexpr std::size_t kComponentTile = 32;

// La griglia e' (colonna di word, blocco di righe): un blocco copre UNA
// posizione di word e kBlockSize righe consecutive.
//
// PERCHE' NON E' PIU' UN THREAD PER WORD CONSECUTIVA
//
// Nella mappatura precedente i thread di un warp coprivano 32 word
// consecutive della stessa riga, quindi per ogni (componente, slot)
// leggevano direction[componente * cols + col] con col = 20*t + slot:
// passo 20 float, 32 indirizzi sparsi su ~40 linee di cache. Con rank 8
// sono 8 * 20 = 160 letture di questo tipo per thread, cioe' ~6400
// transazioni L2 per warp. Con 452 milioni di word per passo e' il costo
// dominante del kernel (misurato 1228 ms/step con Nsight, 24 % del
// passo) — non i 9,05 miliardi di arrotondamenti stocastici, che sono
// aritmetica di registro.
//
// Ora 'firstCol' e' uniforme dentro il blocco: le rank * 20 direzioni
// che servono sono le STESSE per tutti i thread, si caricano una volta
// in memoria condivisa (accessi contigui, quindi coalescenti) e si
// rileggono in broadcast. Il fattore di riuso e' kBlockSize = 256.
//
// Il prezzo: packed[row * wordsPerRow + wordCol] diventa un accesso a
// passo wordsPerRow, quindi non coalescente. Sono 3,8 GiB per passo
// (lettura piu' scrittura) con amplificazione ~8x, cioe' ~30 GiB a 448
// GB/s = ~68 ms STIMATI. L'A/B reale sulla RTX 5060 conferma il
// compromesso, ma non la stima iniziale di circa un secondo risparmiato:
// la mediana dell'intero endStep scende da 512,38 a 324,27 ms (-36,7%)
// e il throughput totale cresce del 2,7%. Dettagli in docs/blackbit.md.
//
// I trit prodotti sono BIT-IDENTICI: 'update' resta accumulato per
// componente crescente, la proiezione e l'arrotondamento stocastico
// dipendono da (riga, colonna) e non dalla forma della griglia. Cambia
// solo il raggruppamento di updateSquared, che alimenta il solo
// updateRms diagnostico.
__global__ void updatePackedKernel(std::uint32_t* packed, float* residual, float latentClamp,
                                   float latentDecay, std::size_t rows, std::size_t cols,
                                   std::size_t wordsPerRow, const float* direction, std::size_t rank,
                                   float learningRate, std::uint64_t seed, std::uint64_t epoch,
                                   std::uint64_t step, DeviceStepStats* stats) {
    const std::size_t wordColumn = blockIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    const std::size_t firstColumn = wordColumn * kTritsPerWord;
    const std::size_t wordIndex = row * wordsPerRow + wordColumn;
    // I thread fuori intervallo non possono uscire subito: partecipano
    // alla riduzione finale con contributi nulli, e soprattutto devono
    // raggiungere le __syncthreads del caricamento in memoria condivisa,
    // altrimenti il blocco si blocca.
    const bool active = row < rows && wordColumn < wordsPerRow;
    unsigned int elements = 0;
    unsigned int flips = 0;
    unsigned int positiveFlips = 0;
    unsigned int negativeFlips = 0;
    unsigned int saturated = 0;
    unsigned int nanInf = 0;
    double updateSquared = 0.0;

    std::uint32_t word = 0;
    int trits[kTritsPerWord];
    float updates[kTritsPerWord];
    for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) updates[slot] = 0.0F;

    if (active) {
        word = packed[wordIndex];
        // projection() dipende da (riga, componente) e NON dalla colonna,
        // ma il ciclo originale la ricalcolava per ogni trit: 20 * rank
        // hash per thread invece di rank. Scambiando i due cicli la
        // proiezione viene valutata una volta per componente e riusata sui
        // 20 trit della word. L'ordine di accumulazione di 'update' resta
        // per componente crescente, esattamente come prima, quindi il
        // risultato e' bit-identico (stessa somma, stessi raggruppamenti).
        for (int byte = 0; byte < 4; ++byte) {
            decodeTritByte(wordByte(word, byte), trits + static_cast<std::size_t>(byte) * kTritsPerByte);
        }
    }

    // Le direzioni delle colonne [firstColumn, firstColumn + 20) sono le
    // stesse per ogni riga, quindi per ogni thread del blocco: si
    // caricano una volta e si rileggono in broadcast dalla memoria
    // condivisa. Le __syncthreads devono stare FUORI da if (active),
    // altrimenti i thread di coda non le raggiungono e il blocco resta
    // fermo.
    __shared__ float sharedDirection[kComponentTile * kTritsPerWord];
    for (std::size_t componentBase = 0; componentBase < rank; componentBase += kComponentTile) {
        const std::size_t chunk = min(kComponentTile, rank - componentBase);
        __syncthreads();
        for (std::size_t index = threadIdx.x; index < chunk * kTritsPerWord; index += blockDim.x) {
            const std::size_t component = componentBase + index / kTritsPerWord;
            const std::size_t column = firstColumn + index % kTritsPerWord;
            sharedDirection[index] = column < cols ? direction[component * cols + column] : 0.0F;
        }
        __syncthreads();
        if (!active) continue;
        for (std::size_t offset = 0; offset < chunk; ++offset) {
            const float scaled =
                -learningRate * projection(seed, epoch, row, componentBase + offset, rank);
            const float* tile = sharedDirection + offset * kTritsPerWord;
            for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
                // La guardia era gia' qui: senza, gli slot di padding
                // dell'ultima word accumulerebbero scaled * 0, che e' 0
                // ma non se scaled fosse non finito.
                if (firstColumn + slot < cols) updates[slot] += scaled * tile[slot];
            }
        }
    }

    if (active) {
    const std::uint64_t stepSeed = splitMix64(seed ^ (step * 0xD1B54A32D192ED03ULL));
    for (std::size_t slot = 0; slot < kTritsPerWord; ++slot) {
        const std::size_t col = firstColumn + slot;
        if (col >= cols) continue;
        const float update = updates[slot];
        if (!isfinite(update)) {
            ++nanInf;
            continue;
        }
        const int oldTrit = trits[slot];
        const std::size_t flat = row * cols + col;
        // Con il residuo il peso latente e' L = trit + residuo: si somma
        // l'aggiornamento a L, si arrotonda al trit piu' vicino, e cio' che
        // avanza resta per il passo dopo. Senza residuo resta il
        // comportamento originale, arrotondamento stocastico e resto
        // scartato.
        int newTrit;
        float target;
        if (residual == nullptr) {
            target = static_cast<float>(oldTrit) + update;
            newTrit = stochasticRoundToTrit(target, stepSeed, flat);
        } else {
            // Il peso latente va limitato a [-1, +1] PRIMA di arrotondare.
            // Senza il limite, un peso che spinge sempre nella stessa
            // direzione fa crescere il residuo senza fine: il trit resta
            // incollato alla saturazione e quel peso smette di imparare, in
            // silenzio e senza che la loss lo segnali subito.
            //
            // Col limite il residuo resta per costruzione in [-0,5, +0,5],
            // che e' anche cio' che ne rende possibile la memorizzazione a
            // pochi bit.
            // Il decadimento agisce sul peso latente PRIMA dell'aggiornamento,
            // come un weight decay disaccoppiato: tira indietro dal bordo i pesi
            // che vi si accumulano, senza interferire con la direzione del
            // gradiente.
            const float precedente = static_cast<float>(oldTrit) + residual[flat];
            const float decaduto = precedente * (1.0F - latentDecay);
            const float latent = fminf(fmaxf(decaduto + update, -latentClamp), latentClamp);
            target = latent;
            const float rounded = rintf(latent);
            newTrit = rounded < -1.0F ? -1 : (rounded > 1.0F ? 1 : static_cast<int>(rounded));
            residual[flat] = latent - static_cast<float>(newTrit);
        }
        trits[slot] = newTrit;
        ++elements;
        updateSquared += static_cast<double>(update) * static_cast<double>(update);
        // Saturazione misurata contro il limite vero, non contro 1. Il
        // criterio originale precede il clamp: allora il latente valeva
        // trit + aggiornamento e superare 1 significava saturare davvero.
        // Con un limite a 2, |latente| >= 1 conta anche i pesi semplicemente
        // decisi, che sono convergenza e non patologia. Quello che serve
        // sapere e' quanti pesi sono incollati al bordo, dove non rispondono
        // piu' al gradiente.
        const float bordo = residual == nullptr ? 1.0F : latentClamp * 0.999F;
        if (target <= -bordo || target >= bordo) ++saturated;
        if (newTrit != oldTrit) {
            ++flips;
            if (newTrit > oldTrit) {
                ++positiveFlips;
            } else {
                ++negativeFlips;
            }
        }
    }
    for (int byte = 0; byte < 4; ++byte) {
        const int* byteTrits = trits + static_cast<std::size_t>(byte) * kTritsPerByte;
        word = setWordByte(word, byte,
                           encodeTritByte(byteTrits[0], byteTrits[1], byteTrits[2], byteTrits[3], byteTrits[4]));
    }
    packed[wordIndex] = word;
    }  // if (active)

    // Un blocco copre al massimo 256 righe * 20 trit = 5120 trit,
    // quindi i contatori stanno in 32 bit anche con la nuova mappatura.
    // Prima ogni thread faceva fino a otto atomiche globali: su tutto il
    // modello sono oltre un miliardo di atomiche per step. Warp shuffle
    // piu' riduzione fra warp lasciano una sola serie di atomiche per
    // blocco. I contatori sono interi, quindi identici; per la somma dei
    // quadrati cambia solo l'ordine di accumulazione di una statistica
    // diagnostica, come gia' fatto per gradientStatsKernel.
    constexpr unsigned int kFullMask = 0xFFFFFFFFU;
    constexpr int kWarpSize = 32;
    constexpr int kWarps = kBlockSize / kWarpSize;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        elements += __shfl_down_sync(kFullMask, elements, offset);
        flips += __shfl_down_sync(kFullMask, flips, offset);
        positiveFlips += __shfl_down_sync(kFullMask, positiveFlips, offset);
        negativeFlips += __shfl_down_sync(kFullMask, negativeFlips, offset);
        saturated += __shfl_down_sync(kFullMask, saturated, offset);
        nanInf += __shfl_down_sync(kFullMask, nanInf, offset);
        updateSquared += __shfl_down_sync(kFullMask, updateSquared, offset);
    }

    __shared__ unsigned int sharedCounters[6][kWarps];
    __shared__ double sharedSquared[kWarps];
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    if (lane == 0) {
        sharedCounters[0][warp] = elements;
        sharedCounters[1][warp] = flips;
        sharedCounters[2][warp] = positiveFlips;
        sharedCounters[3][warp] = negativeFlips;
        sharedCounters[4][warp] = saturated;
        sharedCounters[5][warp] = nanInf;
        sharedSquared[warp] = updateSquared;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    unsigned long long totals[6] = {0, 0, 0, 0, 0, 0};
    double squared = 0.0;
    for (int w = 0; w < kWarps; ++w) {
        for (int counter = 0; counter < 6; ++counter) {
            totals[counter] += sharedCounters[counter][w];
        }
        squared += sharedSquared[w];
    }
    if (totals[0] != 0) {
        atomicAdd(&stats->elements, totals[0]);
        atomicAdd(&stats->updateCount, totals[0]);
        atomicAdd(&stats->updateSquared, squared);
    }
    if (totals[1] != 0) atomicAdd(&stats->flips, totals[1]);
    if (totals[2] != 0) atomicAdd(&stats->positiveFlips, totals[2]);
    if (totals[3] != 0) atomicAdd(&stats->negativeFlips, totals[3]);
    if (totals[4] != 0) atomicAdd(&stats->saturated, totals[4]);
    if (totals[5] != 0) atomicAdd(&stats->nanInf, totals[5]);
}

}  // namespace

LowRankProjectedOptimizer::LowRankProjectedOptimizer(LowRankOptimizerOptions options)
    : options_(options), deviceStats_(sizeof(DeviceStepStats), MemoryArena::Optimizer) {
    if (options_.rank == 0 || options_.projectionInterval == 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: rank and projection interval must be positive");
    }
    BLACKFORGE_CUDA_CHECK(cudaMemset(deviceStats_.data(), 0, deviceStats_.bytes()));
}

void LowRankProjectedOptimizer::allocateState(const std::string& name, State& state,
                                               cuda::TernaryTensor& weight) {
    state.weight = &weight;
    state.rows = weight.rows();
    state.cols = weight.rowLength();
    const auto override = rankOverrides_.find(name);
    state.rank = std::min(override == rankOverrides_.end() ? options_.rank : override->second, state.rows);
    state.seed = options_.seed ^ parameterNameHash(name);
    const std::size_t bytes = state.rank * state.cols * sizeof(float);
    state.firstMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.secondMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.accumulator = Buffer(bytes, MemoryArena::Optimizer);
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.firstMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.secondMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, bytes));
    if (options_.errorFeedback) {
        const std::size_t residualBytes = state.rows * state.cols * sizeof(float);
        state.tritResidual = Buffer(residualBytes, MemoryArena::Optimizer);
        BLACKFORGE_CUDA_CHECK(cudaMemset(state.tritResidual.data(), 0, residualBytes));
    }

    state.scaleCount = state.rows * weight.groupsPerRow();
    const std::size_t scaleBytes = state.scaleCount * sizeof(float);
    state.scaleFirstMoment = Buffer(scaleBytes, MemoryArena::Optimizer);
    state.scaleSecondMoment = Buffer(scaleBytes, MemoryArena::Optimizer);
    state.scaleAccumulator = Buffer(scaleBytes, MemoryArena::Optimizer);
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.scaleFirstMoment.data(), 0, scaleBytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.scaleSecondMoment.data(), 0, scaleBytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.scaleAccumulator.data(), 0, scaleBytes));
    if (state.scaleCount != 0) {
        Buffer squared(sizeof(double), MemoryArena::Temporary);
        BLACKFORGE_CUDA_CHECK(cudaMemset(squared.data(), 0, squared.bytes()));
        squaredNormKernel<<<gridFor(state.scaleCount), kBlockSize>>>(weight.scales(), state.scaleCount,
                                                                     squared.as<double>());
        double hostSquared = 0.0;
        BLACKFORGE_CUDA_CHECK(
            cudaMemcpy(&hostSquared, squared.data(), sizeof(hostSquared), cudaMemcpyDeviceToHost));
        const double rms = std::sqrt(hostSquared / static_cast<double>(state.scaleCount));
        state.scaleRms = rms > 0.0 ? static_cast<float>(rms) : 1.0F;
    }
}

void LowRankProjectedOptimizer::registerTernary(const std::string& name, cuda::TernaryTensor& weight) {
    if (denseStates_.count(name) != 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    auto [iterator, inserted] = states_.try_emplace(name);
    if (!inserted) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    allocateState(name, iterator->second, weight);
}

void LowRankProjectedOptimizer::registerDense(const std::string& name, Tensor& values) {
    if (states_.count(name) != 0 || values.elementCount() == 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: invalid or duplicate dense parameter '" +
                                    name + "'");
    }
    auto [iterator, inserted] = denseStates_.try_emplace(name);
    if (!inserted) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: parameter '" + name + "' already registered");
    }
    DenseState& state = iterator->second;
    state.values = &values;
    const std::size_t bytes = values.bytes();
    state.firstMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.secondMoment = Buffer(bytes, MemoryArena::Optimizer);
    state.accumulator = Buffer(bytes, MemoryArena::Optimizer);
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.firstMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.secondMoment.data(), 0, bytes));
    BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, bytes));
}

void LowRankProjectedOptimizer::setRankFor(const std::string& name, std::size_t rank) {
    if (rank == 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: rank must be positive");
    }
    if (states_.count(name) != 0) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: setRankFor must be called before registration");
    }
    rankOverrides_[name] = rank;
}

LowRankProjectedOptimizer::State& LowRankProjectedOptimizer::stateFor(const ParameterId& id) {
    auto found = states_.find(id.name);
    if (found == states_.end()) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: unregistered parameter '" + id.name + "'");
    }
    if (found->second.rows != id.rows || found->second.cols != id.cols) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: shape mismatch for parameter '" + id.name + "'");
    }
    return found->second;
}

void LowRankProjectedOptimizer::consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow,
                                                            std::size_t rowCount, const float* deviceBlock) {
    // Attribuita a Optimizer anche se avviene DENTRO il backward: la
    // proiezione low-rank e' lavoro dell'ottimizzatore, non del calcolo
    // del gradiente. Si sovrappone quindi alle fasi che la contengono
    // (esperti, attenzione, testa) — voluto: il report dice esplicitamente
    // che le fasi possono sovrapporsi, ed e' l'unico modo di vedere quanto
    // costa la proiezione senza confonderla col GEMM che la precede.
    GpuPhaseScope scope(GpuPhase::Optimizer);
    State& state = stateFor(id);
    if (firstRow + rowCount > state.rows || (deviceBlock == nullptr && rowCount != 0)) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: invalid gradient block");
    }
    state.touched = true;
    // Le statistiche del gradiente sono calcolate dentro la proiezione,
    // dai blocchi con component == 0: coprono ogni elemento esattamente
    // una volta e il valore e' gia' in registro.
    const dim3 projectionGrid(gridFor(state.cols), static_cast<unsigned int>(state.rank));
    projectGradientKernel<<<projectionGrid, kBlockSize>>>(
        state.accumulator.as<float>(), deviceBlock, firstRow, rowCount, state.cols, state.rank, state.seed,
        state.projectionEpoch, deviceStats_.as<DeviceStepStats>());
    const dim3 scaleGrid(static_cast<unsigned int>(state.weight->groupsPerRow()),
                         static_cast<unsigned int>(rowCount));
    scaleGradientKernel<<<scaleGrid, kCudaWarpSizeOptimizer>>>(
        state.scaleAccumulator.as<float>(), deviceBlock, state.weight->packedWords(), firstRow, rowCount,
        state.cols, state.weight->wordsPerRow(), state.weight->groupsPerRow(), state.weight->groupSize());
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
}

void LowRankProjectedOptimizer::consumeDenseGradient(const ParameterId& id, const float* deviceValues,
                                                      std::size_t count) {
    GpuPhaseScope scope(GpuPhase::Optimizer);
    auto found = denseStates_.find(id.name);
    if (found == denseStates_.end()) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: unregistered dense parameter '" +
                                    id.name + "'");
    }
    DenseState& state = found->second;
    if (count != state.values->elementCount() || id.rows * id.cols != count ||
        (deviceValues == nullptr && count != 0)) {
        throw std::invalid_argument("CUDA LowRankProjectedOptimizer: dense gradient shape mismatch for '" +
                                    id.name + "'");
    }
    gradientStatsKernel<<<gridFor(count), kBlockSize>>>(deviceValues, count, deviceStats_.as<DeviceStepStats>());
    accumulateDenseKernel<<<gridFor(count), kBlockSize>>>(state.accumulator.as<float>(), deviceValues, count);
    BLACKFORGE_CUDA_CHECK(cudaGetLastError());
    state.touched = true;
}

void LowRankProjectedOptimizer::endStep() {
    const float bias1 = 1.0F - std::pow(options_.beta1, static_cast<float>(step_ + 1));
    const float bias2 = 1.0F - std::pow(options_.beta2, static_cast<float>(step_ + 1));
    for (auto& [name, state] : states_) {
        (void)name;
        if (!state.touched) continue;
        const std::size_t projectedValues = state.rank * state.cols;
        adamDirectionKernel<<<gridFor(projectedValues), kBlockSize>>>(
            state.firstMoment.as<float>(), state.secondMoment.as<float>(), state.accumulator.as<float>(),
            projectedValues, options_.beta1, options_.beta2, bias1, bias2, options_.eps,
            deviceStats_.as<DeviceStepStats>());
        // (colonna di word, blocco di righe): vedi il commento su
        // updatePackedKernel per il perche' della mappatura.
        const dim3 updateGrid(static_cast<unsigned int>(state.weight->wordsPerRow()),
                              gridFor(state.rows));
        updatePackedKernel<<<updateGrid, kBlockSize>>>(
            state.weight->packedWords(),
            state.tritResidual.bytes() == 0 ? nullptr : state.tritResidual.as<float>(), options_.latentClamp,
            options_.latentDecay, state.rows,
            state.cols, state.weight->wordsPerRow(),
            state.accumulator.as<float>(), state.rank, options_.learningRate, state.seed, state.projectionEpoch,
            step_, deviceStats_.as<DeviceStepStats>());
        BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, state.accumulator.bytes()));
        // Le scale sono parametri continui ordinari: stesso Adam dei tensori
        // densi, senza proiezione e senza quantizzazione. parameterScale a 1
        // perche' qui l'aggiornamento non e' in unita' di griglia ternaria, e
        // weightDecay a 0 perche' far decadere una scala verso zero
        // spegnerebbe il gruppo.
        if (state.scaleCount != 0) {
            denseAdamKernel<<<gridFor(state.scaleCount), kBlockSize>>>(
                state.weight->scales(), state.scaleFirstMoment.as<float>(),
                state.scaleSecondMoment.as<float>(), state.scaleAccumulator.as<float>(), state.scaleCount,
                options_.beta1, options_.beta2, bias1, bias2, options_.eps, options_.learningRate,
                0.0F, state.scaleRms, deviceStats_.as<DeviceStepStats>());
            BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        }
        state.touched = false;
    }
    for (auto& [name, state] : denseStates_) {
        (void)name;
        if (!state.touched) continue;
        Buffer squared(sizeof(double), MemoryArena::Temporary);
        BLACKFORGE_CUDA_CHECK(cudaMemset(squared.data(), 0, squared.bytes()));
        const std::size_t count = state.values->elementCount();
        squaredNormKernel<<<gridFor(count), kBlockSize>>>(state.values->data(), count, squared.as<double>());
        double hostSquared = 0.0;
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(&hostSquared, squared.data(), sizeof(hostSquared), cudaMemcpyDeviceToHost));
        const float parameterRms = static_cast<float>(std::sqrt(hostSquared / static_cast<double>(count)));
        const float scale = parameterRms > 0.0F ? parameterRms : 1.0F;
        denseAdamKernel<<<gridFor(count), kBlockSize>>>(
            state.values->data(), state.firstMoment.as<float>(), state.secondMoment.as<float>(),
            state.accumulator.as<float>(), count, options_.beta1, options_.beta2, bias1, bias2, options_.eps,
            options_.learningRate, options_.weightDecay, scale, deviceStats_.as<DeviceStepStats>());
        BLACKFORGE_CUDA_CHECK(cudaGetLastError());
        state.touched = false;
    }
    BLACKFORGE_CUDA_CHECK(cudaDeviceSynchronize());

    DeviceStepStats device{};
    BLACKFORGE_CUDA_CHECK(cudaMemcpy(&device, deviceStats_.data(), sizeof(device), cudaMemcpyDeviceToHost));
    LowRankOptimizerStepStats last;
    last.elements = static_cast<std::size_t>(device.elements);
    last.flips = static_cast<std::size_t>(device.flips);
    last.positiveFlips = static_cast<std::size_t>(device.positiveFlips);
    last.negativeFlips = static_cast<std::size_t>(device.negativeFlips);
    last.saturatedTrits = static_cast<std::size_t>(device.saturated);
    last.nanInfCount = static_cast<std::size_t>(device.nanInf);
    last.gradientRms = device.gradientCount == 0 ? 0.0 : std::sqrt(device.gradientSquared / device.gradientCount);
    last.optimizerNorm = device.optimizerCount == 0 ? 0.0 : std::sqrt(device.optimizerSquared / device.optimizerCount);
    last.updateRms = device.updateCount == 0 ? 0.0 : std::sqrt(device.updateSquared / device.updateCount);
    stats_.lastStep = last;
    stats_.totalFlips += last.flips;
    stats_.totalElements += last.elements;
    ++step_;
    stats_.stepCount = step_;

    if (step_ % options_.projectionInterval == 0) {
        for (auto& [name, state] : states_) {
            (void)name;
            ++state.projectionEpoch;
            BLACKFORGE_CUDA_CHECK(cudaMemset(state.firstMoment.data(), 0, state.firstMoment.bytes()));
            BLACKFORGE_CUDA_CHECK(cudaMemset(state.secondMoment.data(), 0, state.secondMoment.bytes()));
        }
        ++stats_.projectionReseeds;
    }
    BLACKFORGE_CUDA_CHECK(cudaMemset(deviceStats_.data(), 0, deviceStats_.bytes()));
}

std::size_t LowRankProjectedOptimizer::stateBytes() const {
    std::size_t result = deviceStats_.bytes();
    for (const auto& [name, state] : states_) {
        (void)name;
        result += state.firstMoment.bytes() + state.secondMoment.bytes() + state.accumulator.bytes();
    }
    for (const auto& [name, state] : denseStates_) {
        (void)name;
        result += state.firstMoment.bytes() + state.secondMoment.bytes() + state.accumulator.bytes();
    }
    return result;
}

std::size_t LowRankProjectedOptimizer::conventionalStateBytes() const {
    std::size_t result = 0;
    for (const auto& [name, state] : states_) {
        (void)name;
        result += 2 * state.rows * state.cols * sizeof(float);
    }
    for (const auto& [name, state] : denseStates_) {
        (void)name;
        result += 2 * state.values->bytes();
    }
    return result;
}

namespace {

template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("CUDA optimizer state: truncated stream");
    return value;
}

void writeName(std::ostream& out, const std::string& name) {
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(name.size()));
    out.write(name.data(), static_cast<std::streamsize>(name.size()));
}

std::string readName(std::istream& in) {
    const std::size_t length = readScalar<std::uint32_t>(in);
    std::string name(length, '\0');
    in.read(name.data(), static_cast<std::streamsize>(length));
    if (!in) throw std::runtime_error("CUDA optimizer state: truncated parameter name");
    return name;
}

void writeDeviceVector(std::ostream& out, const Buffer& buffer) {
    const std::size_t count = buffer.bytes() / sizeof(float);
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(count));
    std::vector<float> host(count);
    if (count != 0) {
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(host.data(), buffer.data(), buffer.bytes(), cudaMemcpyDeviceToHost));
        out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(buffer.bytes()));
    }
}

void readDeviceVector(std::istream& in, Buffer& buffer) {
    const std::size_t count = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    if (count * sizeof(float) != buffer.bytes()) {
        throw std::runtime_error("CUDA optimizer state: buffer size differs from registered parameter");
    }
    std::vector<float> host(count);
    if (count != 0) {
        in.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(buffer.bytes()));
        if (!in) throw std::runtime_error("CUDA optimizer state: truncated moment buffer");
        BLACKFORGE_CUDA_CHECK(cudaMemcpy(buffer.data(), host.data(), buffer.bytes(), cudaMemcpyHostToDevice));
    }
}

}  // namespace

void LowRankProjectedOptimizer::serializeState(std::ostream& out) const {
    for (const auto& [name, state] : states_) {
        if (state.touched) throw std::runtime_error("CUDA optimizer state: checkpoint requested mid-step for '" + name + "'");
    }
    for (const auto& [name, state] : denseStates_) {
        if (state.touched) throw std::runtime_error("CUDA optimizer state: checkpoint requested mid-step for '" + name + "'");
    }
    writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(step_));
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(states_.size()));
    for (const auto& [name, state] : states_) {
        writeName(out, name);
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(state.rank));
        writeScalar<std::uint64_t>(out, state.projectionEpoch);
        writeDeviceVector(out, state.firstMoment);
        writeDeviceVector(out, state.secondMoment);
        writeScalar<std::uint64_t>(out, 0);  // CPU-compatible empty consolidation residual.
        // Momenti Adam delle scale: senza, una ripresa da checkpoint
        // ripartirebbe con lo slancio azzerato e non riprodurrebbe il run
        // ininterrotto. Stesso ordine del formato CPU.
        writeDeviceVector(out, state.scaleFirstMoment);
        writeDeviceVector(out, state.scaleSecondMoment);
        writeScalar<float>(out, state.scaleRms);
    }
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(denseStates_.size()));
    for (const auto& [name, state] : denseStates_) {
        writeName(out, name);
        writeDeviceVector(out, state.firstMoment);
        writeDeviceVector(out, state.secondMoment);
    }
    if (!out) throw std::runtime_error("CUDA optimizer state: write failed");
}

namespace {

constexpr char kResiduoTag[8] = {'B', 'F', 'R', 'E', 'S', 'I', 'D', '1'};

// Hash FNV-1a del nome: serve un seme riproducibile fra esecuzioni, e
// std::hash non lo garantisce fra build diverse.
std::uint64_t semeDaNome(const std::string& nome) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : nome) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

}  // namespace

void LowRankProjectedOptimizer::serializeResidual(std::ostream& out) const {
    out.write(kResiduoTag, sizeof(kResiduoTag));
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(states_.size()));
    for (const auto& [name, state] : states_) {
        writeName(out, name);
        writeDeviceVector(out, state.tritResidual);
    }
}

void LowRankProjectedOptimizer::deserializeResidual(std::istream& in) {
    // Il residuo assente NON va sostituito con zero.
    //
    // In regime stazionario il residuo e' L - round(L) per un peso latente che
    // diffonde, quindi distribuito quasi uniformemente su (-0,5, +0,5).
    // Azzerarlo mette tutta la popolazione nello stesso punto: da li' i pesi
    // accumulano insieme e ribaltano in ondate invece che di continuo.
    //
    // Misurato sul riavvio del 2 settembre, che ripartiva da un checkpoint
    // senza residuo: ribaltamenti al 2-3 % contro lo 0,10 % di regime, con un
    // picco 283 passi dopo la ripresa - esattamente i circa 217 passi che
    // servono ad accumulare 0,5 con aggiornamenti da 0,0023 - e la loss salita
    // di 0,18, non ancora rientrata dopo 3000 passi.
    //
    // L'inizializzazione uniforme lascia i trit IDENTICI, perche'
    // l'arrotondamento non cambia, ma restituisce la distribuzione
    // stazionaria. Il modello e' lo stesso, la popolazione e' desincronizzata.
    const std::streampos inizio = in.tellg();
    char tag[sizeof(kResiduoTag)]{};
    in.read(tag, sizeof(tag));
    const bool presente = in.good() && std::memcmp(tag, kResiduoTag, sizeof(tag)) == 0;
    if (!presente) {
        in.clear();
        if (inizio != std::streampos(-1)) in.seekg(inizio);
        for (auto& [name, state] : states_) {
            if (state.tritResidual.bytes() == 0) continue;
            const std::size_t count = state.tritResidual.bytes() / sizeof(float);
            std::vector<float> host(count);
            std::mt19937_64 rng(semeDaNome(name) ^ static_cast<std::uint64_t>(step_));
            std::uniform_real_distribution<float> uniforme(-0.5F, 0.5F);
            for (float& v : host) v = uniforme(rng);
            BLACKFORGE_CUDA_CHECK(cudaMemcpy(state.tritResidual.data(), host.data(),
                                             state.tritResidual.bytes(), cudaMemcpyHostToDevice));
        }
        return;
    }
    const std::size_t count = readScalar<std::uint32_t>(in);
    if (count != states_.size()) throw std::runtime_error("CUDA optimizer residual: conteggio non corrispondente");
    for (std::size_t i = 0; i < count; ++i) {
        const std::string name = readName(in);
        auto found = states_.find(name);
        if (found == states_.end()) throw std::runtime_error("CUDA optimizer residual: parametro sconosciuto '" + name + "'");
        readDeviceVector(in, found->second.tritResidual);
    }
}

void LowRankProjectedOptimizer::deserializeState(std::istream& in) {
    step_ = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
    stats_.stepCount = step_;
    const std::size_t ternaryCount = readScalar<std::uint32_t>(in);
    if (ternaryCount != states_.size()) throw std::runtime_error("CUDA optimizer state: ternary state count mismatch");
    for (std::size_t index = 0; index < ternaryCount; ++index) {
        const std::string name = readName(in);
        auto found = states_.find(name);
        if (found == states_.end()) throw std::runtime_error("CUDA optimizer state: unknown ternary parameter '" + name + "'");
        State& state = found->second;
        const std::size_t rank = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
        if (rank != state.rank) throw std::runtime_error("CUDA optimizer state: rank mismatch for '" + name + "'");
        state.projectionEpoch = readScalar<std::uint64_t>(in);
        readDeviceVector(in, state.firstMoment);
        readDeviceVector(in, state.secondMoment);
        const std::size_t residualCount = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
        if (residualCount != 0) throw std::runtime_error("CUDA optimizer state: consolidation residual is unsupported");
        readDeviceVector(in, state.scaleFirstMoment);
        readDeviceVector(in, state.scaleSecondMoment);
        state.scaleRms = readScalar<float>(in);
        BLACKFORGE_CUDA_CHECK(cudaMemset(state.scaleAccumulator.data(), 0, state.scaleAccumulator.bytes()));
        BLACKFORGE_CUDA_CHECK(cudaMemset(state.accumulator.data(), 0, state.accumulator.bytes()));
        state.touched = false;
    }
    const std::size_t denseCount = readScalar<std::uint32_t>(in);
    if (denseCount != denseStates_.size()) throw std::runtime_error("CUDA optimizer state: dense state count mismatch");
    for (std::size_t index = 0; index < denseCount; ++index) {
        const std::string name = readName(in);
        auto found = denseStates_.find(name);
        if (found == denseStates_.end()) throw std::runtime_error("CUDA optimizer state: unknown dense parameter '" + name + "'");
        readDeviceVector(in, found->second.firstMoment);
        readDeviceVector(in, found->second.secondMoment);
        BLACKFORGE_CUDA_CHECK(cudaMemset(found->second.accumulator.data(), 0, found->second.accumulator.bytes()));
        found->second.touched = false;
    }
}

}  // namespace blackforge::blackbit::cuda
