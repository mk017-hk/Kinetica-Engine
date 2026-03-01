#include "HyperMotion/ml/OnnxInference.h"
#include "HyperMotion/core/Logger.h"
#include <algorithm>
#include <thread>

namespace hm::ml {

// ------------------------------------------------------------------
// Helper: pick a sensible default thread count
// ------------------------------------------------------------------
static int defaultThreadCount() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, hw / 2);
}

#ifdef HM_HAS_ONNXRUNTIME

struct OnnxInference::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "HyperMotion"};
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOpts;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<const char*> inputNamePtrs;
    std::vector<const char*> outputNamePtrs;

    bool loaded = false;
    int requestedThreads = -1;     // -1 = use default
    std::string provider;          // "CPU" or "CUDA"
    std::string loadedModelPath;
};

OnnxInference::OnnxInference() : impl_(std::make_unique<Impl>()) {}
OnnxInference::~OnnxInference() = default;
OnnxInference::OnnxInference(OnnxInference&&) noexcept = default;
OnnxInference& OnnxInference::operator=(OnnxInference&&) noexcept = default;

void OnnxInference::setIntraOpThreads(int threads) {
    impl_->requestedThreads = threads;
}

bool OnnxInference::load(const std::string& modelPath, bool useGPU) {
    try {
        // Thread count: 0 = ORT decides, -1 = half hw concurrency
        int threads = impl_->requestedThreads;
        if (threads < 0)
            threads = defaultThreadCount();
        if (threads > 0)
            impl_->sessionOpts.SetIntraOpNumThreads(threads);

        impl_->sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        impl_->provider = "CPU";

#ifdef HM_USE_CUDA
        if (useGPU) {
            try {
                OrtCUDAProviderOptions cudaOpts{};
                cudaOpts.device_id = 0;
                impl_->sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
                impl_->provider = "CUDA";
            } catch (const Ort::Exception& e) {
                HM_LOG_WARN("OnnxInference",
                    "CUDA provider unavailable for " + modelPath +
                    ", falling back to CPU: " + std::string(e.what()));
                impl_->provider = "CPU";
            }
        }
#else
        if (useGPU) {
            HM_LOG_WARN("OnnxInference",
                "GPU requested but HM_USE_CUDA not enabled at build time; using CPU for: " + modelPath);
        }
#endif

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, modelPath.c_str(), impl_->sessionOpts);

        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < impl_->session->GetInputCount(); ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, alloc);
            impl_->inputNames.emplace_back(name.get());
        }
        for (size_t i = 0; i < impl_->session->GetOutputCount(); ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, alloc);
            impl_->outputNames.emplace_back(name.get());
        }

        impl_->inputNamePtrs.clear();
        for (auto& n : impl_->inputNames) impl_->inputNamePtrs.push_back(n.c_str());
        impl_->outputNamePtrs.clear();
        for (auto& n : impl_->outputNames) impl_->outputNamePtrs.push_back(n.c_str());

        impl_->loaded = true;
        impl_->loadedModelPath = modelPath;
        HM_LOG_INFO("OnnxInference", "Loaded model: " + modelPath +
                     " (" + std::to_string(impl_->inputNames.size()) + " inputs, " +
                     std::to_string(impl_->outputNames.size()) + " outputs, " +
                     "provider=" + impl_->provider +
                     ", threads=" + std::to_string(threads) + ")");
        return true;

    } catch (const Ort::Exception& e) {
        HM_LOG_ERROR("OnnxInference",
            "Failed to load " + modelPath + " (provider=" + impl_->provider + "): " + e.what());
        return false;
    }
}

bool OnnxInference::isLoaded() const { return impl_->loaded; }
const std::string& OnnxInference::activeProvider() const { return impl_->provider; }
const std::string& OnnxInference::modelPath() const { return impl_->loadedModelPath; }

std::vector<Ort::Value> OnnxInference::run(std::vector<Ort::Value>& inputs) {
    return impl_->session->Run(
        Ort::RunOptions{nullptr},
        impl_->inputNamePtrs.data(), inputs.data(), inputs.size(),
        impl_->outputNamePtrs.data(), impl_->outputNamePtrs.size());
}

size_t OnnxInference::numInputs()  const { return impl_->inputNames.size(); }
size_t OnnxInference::numOutputs() const { return impl_->outputNames.size(); }
const std::vector<std::string>& OnnxInference::inputNames()  const { return impl_->inputNames; }
const std::vector<std::string>& OnnxInference::outputNames() const { return impl_->outputNames; }
Ort::MemoryInfo& OnnxInference::memoryInfo() { return impl_->memInfo; }

bool OnnxInference::hasInput(const std::string& name) const {
    return std::find(impl_->inputNames.begin(), impl_->inputNames.end(), name) !=
           impl_->inputNames.end();
}

bool OnnxInference::validateInputCount(size_t expected) const {
    if (impl_->inputNames.size() != expected) {
        HM_LOG_ERROR("OnnxInference",
            "Input count mismatch for " + impl_->loadedModelPath +
            ": expected " + std::to_string(expected) +
            ", got " + std::to_string(impl_->inputNames.size()) +
            " (provider=" + impl_->provider + ")");
        return false;
    }
    return true;
}

#else  // !HM_HAS_ONNXRUNTIME

struct OnnxInference::Impl {
    std::vector<std::string> emptyNames;
    std::string emptyStr;
    bool loaded = false;
};

OnnxInference::OnnxInference() : impl_(std::make_unique<Impl>()) {}
OnnxInference::~OnnxInference() = default;
OnnxInference::OnnxInference(OnnxInference&&) noexcept = default;
OnnxInference& OnnxInference::operator=(OnnxInference&&) noexcept = default;

void OnnxInference::setIntraOpThreads(int /*threads*/) {}

bool OnnxInference::load(const std::string& modelPath, bool /*useGPU*/) {
    HM_LOG_WARN("OnnxInference", "ONNX Runtime not available. Cannot load: " + modelPath);
    return false;
}

bool OnnxInference::isLoaded() const { return false; }
const std::string& OnnxInference::activeProvider() const { return impl_->emptyStr; }
const std::string& OnnxInference::modelPath() const { return impl_->emptyStr; }
size_t OnnxInference::numInputs()  const { return 0; }
size_t OnnxInference::numOutputs() const { return 0; }
const std::vector<std::string>& OnnxInference::inputNames()  const { return impl_->emptyNames; }
const std::vector<std::string>& OnnxInference::outputNames() const { return impl_->emptyNames; }

bool OnnxInference::hasInput(const std::string& /*name*/) const { return false; }
bool OnnxInference::validateInputCount(size_t /*expected*/) const { return false; }

#endif  // HM_HAS_ONNXRUNTIME

} // namespace hm::ml
