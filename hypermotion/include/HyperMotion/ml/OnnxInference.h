#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef HM_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace hm::ml {

/// Lightweight ONNX Runtime session wrapper used by all inference modules.
/// When HM_HAS_ONNXRUNTIME is not defined, all methods are stubs.
class OnnxInference {
public:
    OnnxInference();
    ~OnnxInference();

    OnnxInference(const OnnxInference&) = delete;
    OnnxInference& operator=(const OnnxInference&) = delete;
    OnnxInference(OnnxInference&&) noexcept;
    OnnxInference& operator=(OnnxInference&&) noexcept;

    bool load(const std::string& modelPath, bool useGPU = true);
    bool isLoaded() const;

#ifdef HM_HAS_ONNXRUNTIME
    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs);
    Ort::MemoryInfo& memoryInfo();
#endif

    size_t numInputs()  const;
    size_t numOutputs() const;
    const std::vector<std::string>& inputNames()  const;
    const std::vector<std::string>& outputNames() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
