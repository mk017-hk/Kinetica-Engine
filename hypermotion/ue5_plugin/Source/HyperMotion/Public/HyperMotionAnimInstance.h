#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "HyperMotionComponent.h"
#include "HyperMotionAnimInstance.generated.h"

UCLASS()
class HYPERMOTION_API UHyperMotionAnimInstance : public UAnimInstance
{
    GENERATED_BODY()

public:
    UHyperMotionAnimInstance();

    virtual void NativeInitializeAnimation() override;
    virtual void NativeUpdateAnimation(float DeltaSeconds) override;

    // --- Properties exposed to AnimBP ---

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    float Speed = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    float NormalizedSpeed = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    float Direction = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    float LeanAngle = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    float Fatigue = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    float TransitionAlpha = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    bool bIsMoving = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    bool bIsSprinting = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    bool bIsTurning = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    bool bHasBall = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    EHMMovementState MovementState = EHMMovementState::Idle;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    EHMBallState BallState = EHMBallState::NoBall;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion")
    FPlayerAnimStyle PlayerStyle;

private:
    UPROPERTY()
    TWeakObjectPtr<UHyperMotionComponent> HyperMotionComp;
};
