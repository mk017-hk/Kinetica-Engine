#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "HyperMotionFootIK.generated.h"

USTRUCT(BlueprintType)
struct FFootIKData
{
    GENERATED_BODY()

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    FVector LeftFootOffset = FVector::ZeroVector;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    FVector RightFootOffset = FVector::ZeroVector;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    float HipOffset = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    FRotator LeftFootRotation = FRotator::ZeroRotator;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    FRotator RightFootRotation = FRotator::ZeroRotator;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    bool bLeftFootGrounded = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "FootIK")
    bool bRightFootGrounded = false;
};

UCLASS(ClassGroup=(HyperMotion), meta=(BlueprintSpawnableComponent))
class HYPERMOTION_API UHyperMotionFootIK : public UActorComponent
{
    GENERATED_BODY()

public:
    UHyperMotionFootIK();

    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|FootIK")
    FFootIKData FootIKData;

    // Configuration
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|FootIK|Config")
    float TraceDistance = 100.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|FootIK|Config")
    float FootOffset = 5.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|FootIK|Config")
    float InterpSpeed = 15.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|FootIK|Config")
    FName LeftFootBone = TEXT("foot_l");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|FootIK|Config")
    FName RightFootBone = TEXT("foot_r");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|FootIK|Config")
    TEnumAsByte<ECollisionChannel> TraceChannel = ECC_Visibility;

private:
    struct FFootTraceResult
    {
        FVector ImpactPoint;
        FVector ImpactNormal;
        bool bHit = false;
        float Distance = 0.0f;
    };

    FFootTraceResult TraceFootPosition(const FVector& FootWorldPos);
    FRotator ComputeFootRotation(const FVector& ImpactNormal);

    class USkeletalMeshComponent* MeshComp = nullptr;
    float CurrentHipOffset = 0.0f;
    FVector CurrentLeftFootOffset = FVector::ZeroVector;
    FVector CurrentRightFootOffset = FVector::ZeroVector;
};
