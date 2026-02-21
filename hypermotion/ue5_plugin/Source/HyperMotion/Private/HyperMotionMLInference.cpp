#include "HyperMotionMLInference.h"
#include "Async/Async.h"

#if defined(HM_UE5_ONNX)
#include "onnxruntime_cxx_api.h"
#endif

struct UHyperMotionMLInference::FImplData
{
#if defined(HM_UE5_ONNX)
    TUniquePtr<Ort::Env> OrtEnv;
    TUniquePtr<Ort::Session> OrtSession;
    TUniquePtr<Ort::SessionOptions> SessionOptions;
    Ort::AllocatorWithDefaultOptions Allocator;
    bool bModelLoaded = false;
    bool bUseGPU = false;

    TArray<FString> InputNames;
    TArray<FString> OutputNames;
    TArray<std::vector<int64_t>> InputShapes;
    TArray<std::vector<int64_t>> OutputShapes;
#else
    bool bModelLoaded = false;
    bool bUseGPU = false;
#endif
};

UHyperMotionMLInference::UHyperMotionMLInference()
    : ImplData(MakeUnique<FImplData>())
{
}

UHyperMotionMLInference::~UHyperMotionMLInference() = default;

bool UHyperMotionMLInference::LoadModel(const FString& ModelPath)
{
#if defined(HM_UE5_ONNX)
    try
    {
        ImplData->OrtEnv = MakeUnique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HyperMotion");
        ImplData->SessionOptions = MakeUnique<Ort::SessionOptions>();
        ImplData->SessionOptions->SetIntraOpNumThreads(4);
        ImplData->SessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (ImplData->bUseGPU)
        {
            OrtCUDAProviderOptions CudaOptions;
            CudaOptions.device_id = 0;
            ImplData->SessionOptions->AppendExecutionProvider_CUDA(CudaOptions);
        }

        std::string PathStr = TCHAR_TO_UTF8(*ModelPath);
        ImplData->OrtSession = MakeUnique<Ort::Session>(
            *ImplData->OrtEnv, PathStr.c_str(), *ImplData->SessionOptions);

        // Query input/output info
        size_t NumInputs = ImplData->OrtSession->GetInputCount();
        for (size_t i = 0; i < NumInputs; ++i)
        {
            auto Name = ImplData->OrtSession->GetInputNameAllocated(i, ImplData->Allocator);
            ImplData->InputNames.Add(UTF8_TO_TCHAR(Name.get()));
            auto TypeInfo = ImplData->OrtSession->GetInputTypeInfo(i);
            auto TensorInfo = TypeInfo.GetTensorTypeAndShapeInfo();
            ImplData->InputShapes.Add(TensorInfo.GetShape());
        }

        size_t NumOutputs = ImplData->OrtSession->GetOutputCount();
        for (size_t i = 0; i < NumOutputs; ++i)
        {
            auto Name = ImplData->OrtSession->GetOutputNameAllocated(i, ImplData->Allocator);
            ImplData->OutputNames.Add(UTF8_TO_TCHAR(Name.get()));
            auto TypeInfo = ImplData->OrtSession->GetOutputTypeInfo(i);
            auto TensorInfo = TypeInfo.GetTensorTypeAndShapeInfo();
            ImplData->OutputShapes.Add(TensorInfo.GetShape());
        }

        ImplData->bModelLoaded = true;
        UE_LOG(LogTemp, Log, TEXT("HyperMotion: Loaded ONNX model from %s (%d inputs, %d outputs)"),
               *ModelPath, NumInputs, NumOutputs);
        return true;
    }
    catch (const Ort::Exception& e)
    {
        UE_LOG(LogTemp, Error, TEXT("HyperMotion: ONNX error: %s"), UTF8_TO_TCHAR(e.what()));
        return false;
    }
#else
    UE_LOG(LogTemp, Warning, TEXT("HyperMotion: ONNX Runtime not available. Build with HM_UE5_ONNX=1"));
    return false;
#endif
}

TArray<float> UHyperMotionMLInference::RunInference(
    const TArray<float>& InputData, const TArray<int32>& InputShape)
{
    TArray<float> OutputData;

#if defined(HM_UE5_ONNX)
    if (!ImplData->bModelLoaded || !ImplData->OrtSession)
    {
        UE_LOG(LogTemp, Error, TEXT("HyperMotion: Model not loaded"));
        return OutputData;
    }

    try
    {
        // Build input shape
        std::vector<int64_t> Shape;
        for (int32 S : InputShape) Shape.push_back(static_cast<int64_t>(S));

        auto MemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value InputTensor = Ort::Value::CreateTensor<float>(
            MemoryInfo, const_cast<float*>(InputData.GetData()),
            InputData.Num(), Shape.data(), Shape.size());

        // Build input/output names
        std::vector<const char*> InNames, OutNames;
        std::vector<std::string> InNameStrs, OutNameStrs;
        for (const auto& Name : ImplData->InputNames)
        {
            InNameStrs.push_back(TCHAR_TO_UTF8(*Name));
            InNames.push_back(InNameStrs.back().c_str());
        }
        for (const auto& Name : ImplData->OutputNames)
        {
            OutNameStrs.push_back(TCHAR_TO_UTF8(*Name));
            OutNames.push_back(OutNameStrs.back().c_str());
        }

        auto Results = ImplData->OrtSession->Run(
            Ort::RunOptions{nullptr},
            InNames.data(), &InputTensor, 1,
            OutNames.data(), OutNames.size());

        if (!Results.empty())
        {
            auto& OutTensor = Results[0];
            auto TensorInfo = OutTensor.GetTensorTypeAndShapeInfo();
            size_t NumElements = TensorInfo.GetElementCount();
            const float* OutPtr = OutTensor.GetTensorData<float>();

            OutputData.SetNum(static_cast<int32>(NumElements));
            FMemory::Memcpy(OutputData.GetData(), OutPtr, NumElements * sizeof(float));
        }
    }
    catch (const Ort::Exception& e)
    {
        UE_LOG(LogTemp, Error, TEXT("HyperMotion: Inference error: %s"), UTF8_TO_TCHAR(e.what()));
    }
#endif

    return OutputData;
}

int32 UHyperMotionMLInference::RunInferenceAsync(
    const TArray<float>& InputData, const TArray<int32>& InputShape)
{
    int32 RequestID = NextRequestID++;

    TArray<float> InputCopy = InputData;
    TArray<int32> ShapeCopy = InputShape;

    AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [this, RequestID, InputCopy, ShapeCopy]()
    {
        TArray<float> Result = RunInference(InputCopy, ShapeCopy);

        AsyncTask(ENamedThreads::GameThread, [this, RequestID, Result]()
        {
            OnInferenceComplete.Broadcast(RequestID, Result);
        });
    });

    return RequestID;
}

bool UHyperMotionMLInference::IsModelLoaded() const
{
    return ImplData->bModelLoaded;
}

void UHyperMotionMLInference::SetUseGPU(bool bUseGPU)
{
    ImplData->bUseGPU = bUseGPU;
}
