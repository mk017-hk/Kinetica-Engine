#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

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

    /// Set the number of intra-op threads (call before load).
    /// 0 = let ORT decide, -1 = hardware_concurrency / 2.
    void setIntraOpThreads(int threads);

    bool load(const std::string& modelPath, bool useGPU = true);
    bool isLoaded() const;

    /// Name of the execution provider actually in use ("CPU", "CUDA", or "").
    const std::string& activeProvider() const;

    /// Path of the currently loaded model (empty if none).
    const std::string& modelPath() const;

#ifdef HM_HAS_ONNXRUNTIME
    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs);
    Ort::MemoryInfo& memoryInfo();
#endif

    size_t numInputs()  const;
    size_t numOutputs() const;
    const std::vector<std::string>& inputNames()  const;
    const std::vector<std::string>& outputNames() const;

    // ------------------------------------------------------------------
    // Validation helpers
    // ------------------------------------------------------------------

    /// Check that the model has an input with the given name.
    bool hasInput(const std::string& name) const;

    /// Check that the model's input count matches expected.
    bool validateInputCount(size_t expected) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
