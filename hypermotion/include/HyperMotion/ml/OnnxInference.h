#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace hm::ml {

/// Lightweight ONNX Runtime session wrapper used by all inference modules.
/// Handles session creation, input/output names, and single-shot Run().
class OnnxInference {
public:
    OnnxInference();
    ~OnnxInference();

    OnnxInference(const OnnxInference&) = delete;
    OnnxInference& operator=(const OnnxInference&) = delete;
    OnnxInference(OnnxInference&&) noexcept;
    OnnxInference& operator=(OnnxInference&&) noexcept;

    /// Load an ONNX model. Returns false on failure.
    /// @param useGPU  Attempt CUDA execution provider; falls back to CPU.
    bool load(const std::string& modelPath, bool useGPU = true);
    bool isLoaded() const;

    /// Run inference with pre-built Ort::Value vectors.
    /// Caller owns the input tensors; output tensors are returned.
    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs);

    /// Convenience: number of inputs / outputs and their names.
    size_t numInputs()  const;
    size_t numOutputs() const;
    const std::vector<std::string>& inputNames()  const;
    const std::vector<std::string>& outputNames() const;

    /// Access the allocator (needed to create tensors on the same device).
    Ort::MemoryInfo& memoryInfo();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
