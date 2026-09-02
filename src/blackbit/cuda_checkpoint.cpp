#include "blackforge/blackbit/cuda_checkpoint.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "blackforge/backend/cuda/cuda_check.hpp"

namespace blackforge::blackbit::cuda {

namespace {

constexpr char kMagic[8] = {'B', 'F', 'B', 'I', 'T', '\0', '\0', '\0'};
enum class ParameterKind : std::uint8_t { Ternary = 0, Dense = 1 };

template <typename T>
void writeScalar(std::ostream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("CUDA BlackBit checkpoint: truncated file");
    return value;
}

void writeString(std::ostream& out, const std::string& value) {
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string readString(std::istream& in) {
    const std::size_t length = readScalar<std::uint32_t>(in);
    std::string value(length, '\0');
    in.read(value.data(), static_cast<std::streamsize>(length));
    if (!in) throw std::runtime_error("CUDA BlackBit checkpoint: truncated string");
    return value;
}

struct ParameterCollector {
    std::vector<std::pair<std::string, cuda::TernaryTensor*>> ternary;
    std::vector<std::pair<std::string, Tensor*>> dense;
    void registerTernary(const std::string& name, cuda::TernaryTensor& weight) {
        ternary.emplace_back(name, &weight);
    }
    void registerDense(const std::string& name, Tensor& values) { dense.emplace_back(name, &values); }
};

ParameterCollector collect(BlackBitModel& model) {
    ParameterCollector result;
    model.registerParameters(result);
    return result;
}

void readMagic(std::istream& in, const std::string& path) {
    char magic[sizeof(kMagic)]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("CUDA BlackBit checkpoint: '" + path + "' has invalid magic");
    }
    const std::uint32_t version = readScalar<std::uint32_t>(in);
    if (version != kBlackBitCheckpointVersion) {
        throw std::runtime_error("CUDA BlackBit checkpoint: unsupported version " + std::to_string(version));
    }
}

}  // namespace

void saveCheckpoint(const std::string& path, BlackBitModel& model,
                    const BlackBitTrainingState& state, LowRankProjectedOptimizer* optimizer) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("CUDA BlackBit checkpoint: cannot write '" + path + "'");
    out.write(kMagic, sizeof(kMagic));
    writeScalar<std::uint32_t>(out, kBlackBitCheckpointVersion);
    writeString(out, toJson(model.config()));
    writeScalar<std::uint64_t>(out, state.step);
    writeScalar<std::uint64_t>(out, state.tokensSeen);
    writeScalar<float>(out, state.learningRate);
    writeScalar<std::uint64_t>(out, state.rngSeed);
    writeScalar<std::uint64_t>(out, state.optimizerStep);
    writeScalar<std::uint8_t>(out, optimizer == nullptr ? 0 : 1);
    const ParameterCollector parameters = collect(model);
    writeScalar<std::uint32_t>(out, static_cast<std::uint32_t>(parameters.ternary.size() + parameters.dense.size()));
    for (const auto& [name, weight] : parameters.ternary) {
        writeString(out, name);
        writeScalar<std::uint8_t>(out, static_cast<std::uint8_t>(ParameterKind::Ternary));
        const blackforge::blackbit::TernaryTensor host = weight->download();
        host.serialize(out);
    }
    for (const auto& [name, values] : parameters.dense) {
        writeString(out, name);
        writeScalar<std::uint8_t>(out, static_cast<std::uint8_t>(ParameterKind::Dense));
        const runtime::Tensor host = values->toHost();
        writeScalar<std::uint64_t>(out, static_cast<std::uint64_t>(host.elementCount()));
        out.write(reinterpret_cast<const char*>(host.data().data()),
                  static_cast<std::streamsize>(host.elementCount() * sizeof(float)));
    }
    if (optimizer != nullptr) {
        optimizer->serializeState(out);
        optimizer->serializeResidual(out);
    }
    if (!out) throw std::runtime_error("CUDA BlackBit checkpoint: write failed for '" + path + "'");
}

BlackBitTrainingState loadCheckpoint(const std::string& path, BlackBitModel& model,
                                     LowRankProjectedOptimizer* optimizer) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("CUDA BlackBit checkpoint: cannot read '" + path + "'");
    readMagic(in, path);
    const std::string savedConfig = readString(in);
    if (savedConfig != toJson(model.config())) {
        throw std::runtime_error("CUDA BlackBit checkpoint: model configuration mismatch");
    }
    BlackBitTrainingState state;
    state.step = readScalar<std::uint64_t>(in);
    state.tokensSeen = readScalar<std::uint64_t>(in);
    state.learningRate = readScalar<float>(in);
    state.rngSeed = readScalar<std::uint64_t>(in);
    state.optimizerStep = readScalar<std::uint64_t>(in);
    const bool hasOptimizer = readScalar<std::uint8_t>(in) != 0;
    ParameterCollector parameters = collect(model);
    std::unordered_map<std::string, cuda::TernaryTensor*> ternary;
    std::unordered_map<std::string, Tensor*> dense;
    for (const auto& entry : parameters.ternary) ternary.emplace(entry.first, entry.second);
    for (const auto& entry : parameters.dense) dense.emplace(entry.first, entry.second);
    const std::size_t parameterCount = readScalar<std::uint32_t>(in);
    if (parameterCount != ternary.size() + dense.size()) {
        throw std::runtime_error("CUDA BlackBit checkpoint: parameter count mismatch");
    }
    for (std::size_t index = 0; index < parameterCount; ++index) {
        const std::string name = readString(in);
        const auto kind = static_cast<ParameterKind>(readScalar<std::uint8_t>(in));
        if (kind == ParameterKind::Ternary) {
            blackforge::blackbit::TernaryTensor host = blackforge::blackbit::TernaryTensor::deserialize(in);
            auto found = ternary.find(name);
            if (found == ternary.end() || host.shape() != found->second->shape() ||
                host.groupSize() != found->second->groupSize()) {
                throw std::runtime_error("CUDA BlackBit checkpoint: ternary parameter mismatch for '" + name + "'");
            }
            *found->second = cuda::TernaryTensor(host);
        } else if (kind == ParameterKind::Dense) {
            const std::size_t count = static_cast<std::size_t>(readScalar<std::uint64_t>(in));
            auto found = dense.find(name);
            if (found == dense.end() || count != found->second->elementCount()) {
                throw std::runtime_error("CUDA BlackBit checkpoint: dense parameter mismatch for '" + name + "'");
            }
            std::vector<float> host(count);
            in.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(count * sizeof(float)));
            if (!in) throw std::runtime_error("CUDA BlackBit checkpoint: truncated dense parameter");
            BLACKFORGE_CUDA_CHECK(cudaMemcpy(found->second->data(), host.data(), count * sizeof(float),
                                             cudaMemcpyHostToDevice));
        } else {
            throw std::runtime_error("CUDA BlackBit checkpoint: unknown parameter kind");
        }
    }
    if (hasOptimizer && optimizer != nullptr) {
        optimizer->deserializeState(in);
        optimizer->deserializeResidual(in);
    } else if (!hasOptimizer && optimizer != nullptr) {
        throw std::runtime_error("CUDA BlackBit checkpoint: optimizer state requested but absent");
    }
    return state;
}

}  // namespace blackforge::blackbit::cuda
