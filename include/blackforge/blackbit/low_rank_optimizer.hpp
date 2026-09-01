#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "blackforge/blackbit/gradient.hpp"
#include "blackforge/blackbit/ternary.hpp"

// Ottimizzatore a stato proiettato low-rank per BlackBit (requisiti 7,
// 8 e 9).
//
// IL VINCOLO
//
// AdamW ordinario tiene due momenti float32 per parametro: su
// BlackBit-9B sono 9,05 G * 8 B = 72 GB. Anche in BF16 sarebbero 36 GB.
// Non e' una questione di ottimizzare: e' che l'ottimizzatore da solo
// occupa nove volte la VRAM disponibile.
//
// L'IDEA (GaLore/Q-GaLore, riscritta per questo motore)
//
// Il gradiente di una matrice grande e' quasi di rango basso. Invece di
// tenere lo stato su W in R^(m x n), lo si tiene sulla PROIEZIONE
//
//     R = P^T G,        P in R^(m x r),  R in R^(r x n)
//
// e si ricostruisce l'aggiornamento come dW = P * Adam(R). Lo stato
// passa da 2mn a 2rn: per una matrice di esperto (3072 x 3968) con
// r = 32, da 24,4 M valori a 254 K, cioe' 96 volte meno.
//
// DUE DIFFERENZE DELIBERATE RISPETTO A GaLore
//
// 1. **La proiezione non e' l'SVD del gradiente, e' casuale.** GaLore
//    calcola periodicamente l'SVD di G per prendere il sottospazio
//    dominante — ma per farlo deve avere G INTERO in memoria, che per
//    l'embedding di BlackBit-9B significa 805 MB, esattamente il buffer
//    che questo progetto esiste per non allocare. Qui P e' una matrice
//    di Rademacher (+-1/sqrt(r)) generata al volo dal generatore a
//    contatore: costa ZERO byte, si valuta per riga, e per il lemma di
//    Johnson-Lindenstrauss conserva le distanze del sottospazio con
//    errore controllato. Il sottospazio viene RISEMINATO ogni
//    'projectionInterval' passi, cosi' direzioni diverse vengono
//    esplorate nel tempo invece di restare per sempre fuori portata.
//    E' un sottospazio peggiore di quello dell'SVD; e' l'unico
//    compatibile con un gradiente che non esiste mai per intero.
//
// 2. **L'aggiornamento e' in unita' di griglia.** Adam produce per
//    costruzione valori di modulo ~1, e P ha norma di riga ~1, quindi
//    dW = P * Adam(R) ha modulo ~1: il learning rate misura
//    direttamente la frazione di passo ternario percorsa. Su pesi
//    quantizzati e' l'unica scala che abbia senso (vedi
//    ternary_update.hpp).
//
// PERCHE' LA PROIEZIONE SI ACCUMULA A BLOCCHI
//
// P^T G = somma sui blocchi di righe di (P_blocco^T G_blocco): ogni
// blocco di gradiente che arriva dal backward in streaming contribuisce
// e viene subito buttato. Non serve mai avere G intero — che e'
// precisamente il motivo per cui questa proiezione, e non l'SVD, e'
// compatibile con il requisito 6.

namespace blackforge::blackbit {

struct LowRankOptimizerOptions {
    float learningRate = 0.02F;  // in unita' di griglia ternaria
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float eps = 1e-8F;

    // Decadimento disaccoppiato, applicato SOLO ai parametri densi: un
    // peso ternario e' gia' limitato a [-scala, +scala], contrarlo
    // ulteriormente non ha significato.
    float weightDecay = 0.0F;

    // Accumulo del residuo di arrotondamento ("error feedback").
    //
    // Senza, un aggiornamento piu' piccolo del passo della griglia ternaria
    // viene giocato a dadi e quasi sempre perso: il segnale non arriva mai
    // nei pesi. Con l'accumulo il peso latente e' L = trit + residuo, si
    // aggiorna L, si arrotonda al trit piu' vicino e cio' che avanza resta
    // per il passo successivo, finche' non basta a far scattare un trit.
    //
    // Quando e' attivo l'arrotondamento diventa deterministico: quello
    // stocastico esiste per essere non distorto SENZA memoria, e sommato a
    // una memoria aggiungerebbe rumore che la memoria stessa deve poi
    // correggere.
    //
    // Costa un float per peso. Sul modello 60 M sono 240 MB; su un 9B
    // sarebbero 36 GB, cioe' la copia master che BlackBit esiste per non
    // avere: a quella scala il residuo va compresso a 4 o 8 bit. Qui serve
    // a verificare l'ipotesi dove la memoria non e' un problema.
    bool errorFeedback = false;

    // Rango di default del sottospazio. Configurabile per parametro con
    // setRankFor().
    std::size_t rank = 32;

    // Ogni quanti passi il sottospazio casuale viene riseminato. A
    // riseminatura i momenti vengono azzerati: si riferiscono a una
    // base diversa, e conservarli mescolerebbe coordinate che non
    // hanno piu' lo stesso significato.
    std::size_t projectionInterval = 200;

    std::uint64_t seed = 0xB1ACB17ULL;
};

// Esperimento del requisito 9: W = T + Delta, con Delta = P * residuo
// (basso rango) che accumula la plasticita' e viene periodicamente
// CONSOLIDATO in flip di T.
//
// Rispetto all'aggiornamento diretto (un flip stocastico ad ogni passo)
// riduce la varianza: le spinte piccole si sommano nel residuo finche'
// non valgono un movimento vero, invece di essere giocate a testa o
// croce ogni volta. In cambio introduce uno stato in piu' (il residuo,
// r x n per matrice) e un iperparametro delicato (ogni quanto
// consolidare).
//
// SPERIMENTALE e disattivato per default: il percorso ordinario
// funziona gia' senza. Va acceso deliberatamente.
struct TernaryConsolidationOptions {
    bool enabled = false;
    std::size_t consolidationInterval = 50;

    // Quanto deve valere |Delta| (in unita' di griglia) perche' un trit
    // sia candidato al movimento.
    float flipThreshold = 0.5F;

    // Frazione massima di trit che una singola consolidazione puo'
    // muovere: un tetto contro una consolidazione che ribalti mezza
    // matrice in un colpo.
    float maxFlipFraction = 0.02F;

    // Quanto del residuo sopravvive alla consolidazione (0 = si azzera
    // completamente, 1 = resta tutto).
    float residualDecay = 0.0F;

    // Se true la soglia e' probabilistica (arrotondamento stocastico
    // sul residuo) invece che netta.
    bool stochasticFlip = true;
};

// Statistiche per capire cosa sta facendo l'ottimizzatore.
struct LowRankOptimizerStats {
    std::size_t stepCount = 0;
    std::size_t ternaryFlips = 0;
    std::size_t ternaryElements = 0;
    std::size_t projectionReseeds = 0;
    std::size_t consolidations = 0;

    // Norma del residuo di consolidamento, se attivo: se cresce senza
    // mai scaricarsi, la consolidazione non sta avvenendo.
    double residualNorm = 0.0;

    [[nodiscard]] double flipFraction() const {
        return ternaryElements == 0 ? 0.0 : static_cast<double>(ternaryFlips) / static_cast<double>(ternaryElements);
    }
};

class LowRankProjectedOptimizer : public GradientSink {
public:
    explicit LowRankProjectedOptimizer(LowRankOptimizerOptions options = {},
                                        TernaryConsolidationOptions consolidation = {});
    ~LowRankProjectedOptimizer() override;

    LowRankProjectedOptimizer(const LowRankProjectedOptimizer&) = delete;
    LowRankProjectedOptimizer& operator=(const LowRankProjectedOptimizer&) = delete;

    void registerTernary(const std::string& name, TernaryTensor& weight);
    void registerDense(const std::string& name, std::vector<float>& values);

    // Rango specifico per un parametro, da chiamare PRIMA del primo
    // passo (requisito 7: rango configurabile per layer).
    void setRankFor(const std::string& name, std::size_t rank);

    void consumeWeightGradientBlock(const ParameterId& id, std::size_t firstRow, std::size_t rowCount,
                                     const float* block) override;
    void consumeDenseGradient(const ParameterId& id, const float* values, std::size_t count) override;

    // Applica gli aggiornamenti accumulati e avanza il passo. Va
    // chiamata una volta per ogni passo di addestramento, dopo il
    // backward.
    void endStep();

    void setLearningRate(float learningRate) { options_.learningRate = learningRate; }
    [[nodiscard]] float learningRate() const { return options_.learningRate; }

    // Byte di stato REALMENTE posseduti, non stimati: e' il numero che
    // il benchmark riporta come "optimizer memory".
    [[nodiscard]] std::size_t stateBytes() const;

    // Byte che AdamW ordinario userebbe per gli stessi parametri, per
    // confronto.
    [[nodiscard]] std::size_t conventionalStateBytes() const;

    [[nodiscard]] const LowRankOptimizerStats& stats() const { return stats_; }
    void resetStats();

    [[nodiscard]] const TernaryConsolidationOptions& consolidation() const { return consolidation_; }

    // Serializzazione dello stato per il checkpoint (vedi
    // blackbit/checkpoint.hpp). Non scrive magic ne' versione: e' un
    // blocco componibile, versionato dal contenitore.
    void serializeState(std::ostream& out) const;

    // Lancia std::runtime_error se lo stream finisce prima del previsto
    // o se un nome/forma non corrisponde a un parametro registrato: uno
    // stato dell'ottimizzatore che non combacia col modello e' un
    // errore, non qualcosa da caricare a meta'.
    void deserializeState(std::istream& in);

    [[nodiscard]] std::size_t stepCount() const { return step_; }
    void setStepCount(std::size_t step) { step_ = step; }

private:
    struct TernaryState {
        TernaryTensor* weight = nullptr;
        std::size_t rank = 0;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::uint64_t seed = 0;             // identita' del parametro, fissa
        std::uint64_t projectionEpoch = 0;  // cambia a ogni riseminatura

        std::vector<float> firstMoment;   // rank x cols
        std::vector<float> secondMoment;  // rank x cols
        std::vector<float> accumulator;   // rank x cols, azzerato a ogni passo
        std::vector<float> residual;      // rank x cols, solo se la consolidazione e' attiva

        // Le scale per gruppo sono l'unica parte continua di un peso
        // ternario. Senza il loro gradiente ogni peso resta inchiodato ai
        // tre valori decisi all'inizializzazione: e' esattamente cio' che
        // impediva al modello di adattarsi ai dati.
        std::vector<float> scaleFirstMoment;   // rows x groupsPerRow
        std::vector<float> scaleSecondMoment;  // rows x groupsPerRow
        std::vector<float> scaleAccumulator;   // rows x groupsPerRow
        float scaleRms = 1.0F;
        bool touched = false;
    };

    // Dichiarata dopo TernaryState: prima della definizione, 'struct
    // TernaryState' nella firma avrebbe introdotto un tipo nuovo invece di
    // riferirsi a quello annidato.
    void applyScaleUpdate(TernaryState& state);

    struct DenseState {
        std::vector<float>* values = nullptr;
        std::vector<float> firstMoment;
        std::vector<float> secondMoment;
        std::vector<float> accumulator;
        bool touched = false;
    };

    TernaryState& ternaryStateFor(const ParameterId& id);
    void applyTernaryUpdate(TernaryState& state);
    void applyDenseUpdate(DenseState& state);
    void consolidate(TernaryState& state);
    void accountState(std::ptrdiff_t deltaBytes);

    LowRankOptimizerOptions options_;
    TernaryConsolidationOptions consolidation_;
    LowRankOptimizerStats stats_;
    std::size_t step_ = 0;
    std::size_t accountedBytes_ = 0;
    std::unordered_map<std::string, std::size_t> rankOverrides_;
    std::unordered_map<std::string, TernaryState> ternary_;
    std::unordered_map<std::string, DenseState> dense_;
};

// Elemento (row, component) della matrice di proiezione: Rademacher
// scalata, generata al volo dal seme invece che memorizzata. E'
// esposta per i test, che verificano che la stessa coppia produca
// sempre lo stesso valore e che le colonne siano quasi ortogonali.
float projectionEntry(std::uint64_t seed, std::uint64_t epoch, std::size_t row, std::size_t component,
                       std::size_t rank);

}  // namespace blackforge::blackbit
