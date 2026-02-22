#include "HyperMotion/ml/OnnxInference.h"
#include "HyperMotion/core/Logger.h"
#include <algorithm>

namespace hm::ml {

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
};

OnnxInference::OnnxInference() : impl_(std::make_unique<Impl>()) {}
OnnxInference::~OnnxInference() = default;
OnnxInference::OnnxInference(OnnxInference&&) noexcept = default;
OnnxInference& OnnxInference::operator=(OnnxInference&&) noexcept = default;

bool OnnxInference::load(const std::string& modelPath, bool useGPU) {
    try {
        impl_->sessionOpts.SetIntraOpNumThreads(4);
        impl_->sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef HM_USE_CUDA
        if (useGPU) {
            OrtCUDAProviderOptions cudaOpts{};
            cudaOpts.device_id = 0;
            impl_->sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
        }
#endif

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, modelPath.c_str(), impl_->sessionOpts);

        // Cache input/output names
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < impl_->session->GetInputCount(); ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, alloc);
            impl_->inputNames.emplace_back(name.get());
        }
        for (size_t i = 0; i < impl_->session->GetOutputCount(); ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, alloc);
            impl_->outputNames.emplace_back(name.get());
        }

        // Build const char* arrays for Run()
        impl_->inputNamePtrs.clear();
        for (auto& n : impl_->inputNames) impl_->inputNamePtrs.push_back(n.c_str());
        impl_->outputNamePtrs.clear();
        for (auto& n : impl_->outputNames) impl_->outputNamePtrs.push_back(n.c_str());

        impl_->loaded = true;
        HM_LOG_INFO("OnnxInference", "Loaded model: " + modelPath +
                     " (" + std::to_string(impl_->inputNames.size()) + " inputs, " +
                     std::to_string(impl_->outputNames.size()) + " outputs)");
        return true;

    } catch (const Ort::Exception& e) {
        HM_LOG_ERROR("OnnxInference", "Failed to load " + modelPath + ": " + e.what());
        return false;
    }
}

bool OnnxInference::isLoaded() const { return impl_->loaded; }

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

} // namespace hm::ml
