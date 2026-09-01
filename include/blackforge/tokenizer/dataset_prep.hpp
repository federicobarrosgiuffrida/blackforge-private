#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "blackforge/tokenizer/tokenizer.hpp"

namespace blackforge::tokenizer {

// Costruisce un dataset .bfdata pronto per l'addestramento di un
// modello linguistico (next-token-prediction) a partire da un corpus
// di testo grezzo: tokenizza con 'tok', poi affetta lo stream di id
// risultante in finestre non sovrapposte di 'seqLen' token (stride ==
// seqLen; l'ultima finestra, se il corpus tokenizzato non basta a
// riempirla, viene scartata — non e' un errore, a meno che NESSUNA
// finestra sia completa). Ogni esempio prodotto ha input `[seqLen]`
// (id di token) e target **sparso** `[seqLen]` (l'indice del token
// successivo per ogni posizione, shift-by-one causale — non un
// vettore one-hot denso `[seqLen, vocabSize]`: per un vocabolario di
// decine di migliaia di token, il target denso sprecherebbe
// `vocabSize` volte più memoria del necessario). Pensato per essere
// allenato con `loss cross_entropy_sparse` (vedi
// backend::{cpu,cuda}::softmaxCrossEntropySparse), NON `cross_entropy`
// (che si aspetta un target denso della stessa forma dei logit).
//
// Ritorna il numero di esempi scritti. Lancia std::invalid_argument
// se il corpus tokenizzato non produce nemmeno una finestra completa
// (serve almeno seqLen + 1 token, l'ultimo serve come target
// dell'ultima posizione della finestra).
//
// NOTA: come data::Dataset in generale, l'intero dataset risultante
// viene costruito in memoria prima di essere scritto su disco — per
// corpora molto grandi e' un limite esplicito, che la milestone di
// caricamento dati efficiente (streaming/shard) rimuovera'; questa
// funzione non e' pensata per corpora da gigabyte cosi' come e' oggi.
// Variante che riceve i token gia' codificati, per i corpus troppo grandi
// da tenere in memoria come stringa.
std::size_t buildLanguageModelDatasetFromIds(const std::vector<std::uint32_t>& ids, std::size_t seqLen,
                                              const std::string& outputPath);

std::size_t buildLanguageModelDataset(const Tokenizer& tok, const std::string& corpus, std::size_t seqLen,
                                       const std::string& outputPath);

// Costruisce un dataset .bfdata pronto per l'addestramento di un
// modello linguistico MASCHERATO (MLM, stile BERT) a partire da un
// corpus di testo grezzo: tokenizza con 'tok', affetta lo stream di id
// risultante in finestre non sovrapposte di 'seqLen' token (nessuno
// shift-by-one, a differenza di buildLanguageModelDataset: qui il
// compito e' ricostruire i token ORIGINALI di UNA sequenza, non
// predire il token successivo), poi per ogni posizione di ogni finestra
// decide, con probabilita' 'maskProb' e seminato deterministicamente da
// 'seed', se mascherarla: se si', l'input in quella posizione diventa
// l'id di pad del tokenizer (Tokenizer::kPadId, riusato come token
// <mask> — encode() non lo produce mai su testo reale, quindi non c'e'
// ambiguita') e il target e' l'id del token ORIGINALE; se no, l'input
// resta il token originale e il target e' -1 (ignora questa posizione,
// vedi backend::cpu::softmaxCrossEntropyMasked). Pensato per essere
// allenato con `loss cross_entropy_masked`.
//
// Ritorna il numero di esempi scritti. Lancia std::invalid_argument se
// il corpus tokenizzato non produce nemmeno una finestra completa (serve
// almeno seqLen token) o se maskProb non e' in (0, 1].
std::size_t buildMaskedLanguageModelDataset(const Tokenizer& tok, const std::string& corpus, std::size_t seqLen,
                                             float maskProb, unsigned int seed, const std::string& outputPath);

}  // namespace blackforge::tokenizer
