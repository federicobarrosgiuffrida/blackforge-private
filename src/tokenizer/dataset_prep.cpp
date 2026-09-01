#include "blackforge/tokenizer/dataset_prep.hpp"

#include <random>
#include <stdexcept>

#include "blackforge/data/dataset.hpp"

namespace blackforge::tokenizer {

std::size_t buildLanguageModelDataset(const Tokenizer& tok, const std::string& corpus, std::size_t seqLen,
                                       const std::string& outputPath) {
    return buildLanguageModelDatasetFromIds(tok.encode(corpus), seqLen, outputPath);
}

std::size_t buildLanguageModelDatasetFromIds(const std::vector<std::uint32_t>& ids, std::size_t seqLen,
                                              const std::string& outputPath) {
    if (ids.size() < seqLen + 1) {
        throw std::invalid_argument("buildLanguageModelDatasetFromIds: il corpus tokenizzato ha solo " +
                                     std::to_string(ids.size()) + " token, ne servono almeno " +
                                     std::to_string(seqLen + 1) + " (seqLen + 1) per una singola finestra");
    }

    std::size_t numExamples = (ids.size() - 1) / seqLen;

    std::vector<float> inputs;
    std::vector<float> targets;
    inputs.reserve(numExamples * seqLen);
    targets.reserve(numExamples * seqLen);

    for (std::size_t example = 0; example < numExamples; ++example) {
        std::size_t windowStart = example * seqLen;
        for (std::size_t s = 0; s < seqLen; ++s) {
            inputs.push_back(static_cast<float>(ids[windowStart + s]));
            targets.push_back(static_cast<float>(ids[windowStart + s + 1]));
        }
    }

    data::saveDataset(outputPath, {seqLen}, {seqLen}, inputs, targets, numExamples);
    return numExamples;
}

std::size_t buildMaskedLanguageModelDataset(const Tokenizer& tok, const std::string& corpus, std::size_t seqLen,
                                             float maskProb, unsigned int seed, const std::string& outputPath) {
    if (maskProb <= 0.0F || maskProb > 1.0F) {
        throw std::invalid_argument("buildMaskedLanguageModelDataset: maskProb deve essere in (0, 1], trovato " +
                                     std::to_string(maskProb));
    }

    std::vector<std::uint32_t> ids = tok.encode(corpus);
    if (ids.size() < seqLen) {
        throw std::invalid_argument("buildMaskedLanguageModelDataset: il corpus tokenizzato ha solo " +
                                     std::to_string(ids.size()) + " token, ne servono almeno " +
                                     std::to_string(seqLen));
    }

    std::size_t numExamples = ids.size() / seqLen;

    std::vector<float> inputs;
    std::vector<float> targets;
    inputs.reserve(numExamples * seqLen);
    targets.reserve(numExamples * seqLen);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    for (std::size_t example = 0; example < numExamples; ++example) {
        std::size_t windowStart = example * seqLen;
        for (std::size_t s = 0; s < seqLen; ++s) {
            std::uint32_t originalToken = ids[windowStart + s];
            if (dist(rng) < maskProb) {
                inputs.push_back(static_cast<float>(Tokenizer::kPadId));
                targets.push_back(static_cast<float>(originalToken));
            } else {
                inputs.push_back(static_cast<float>(originalToken));
                targets.push_back(-1.0F);
            }
        }
    }

    data::saveDataset(outputPath, {seqLen}, {seqLen}, inputs, targets, numExamples);
    return numExamples;
}

}  // namespace blackforge::tokenizer
