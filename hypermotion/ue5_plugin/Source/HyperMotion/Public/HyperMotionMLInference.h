#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "HyperMotionMLInference.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnInferenceComplete, int32, RequestID, const TArray<float>&, OutputData);

UCLASS(BlueprintType)
class HYPERMOTION_API UHyperMotionMLInference : public UObject
{
    GENERATED_BODY()

public:
    UHyperMotionMLInference();
    ~UHyperMotionMLInference();

    // Load ONNX model
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|ML")
    bool LoadModel(const FString& ModelPath);

    // Synchronous inference
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|ML")
    TArray<float> RunInference(const TArray<float>& InputData,
                                const TArray<int32>& InputShape);

    // Asynchronous inference
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|ML")
    int32 RunInferenceAsync(const TArray<float>& InputData,
                             const TArray<int32>& InputShape);

    UPROPERTY(BlueprintAssignable, Category = "HyperMotion|ML")
    FOnInferenceComplete OnInferenceComplete;

    UFUNCTION(BlueprintPure, Category = "HyperMotion|ML")
    bool IsModelLoaded() const;

    UFUNCTION(BlueprintCallable, Category = "HyperMotion|ML")
    void SetUseGPU(bool bUseGPU);

private:
    struct FImplData;
    TUniquePtr<FImplData> ImplData;

    int32 NextRequestID = 0;
};
