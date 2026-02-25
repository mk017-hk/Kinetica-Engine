#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "HyperMotionComponent.generated.h"

// Movement states
UENUM(BlueprintType)
enum class EHMMovementState : uint8
{
    Idle,
    Walk,
    Jog,
    Run,
    Sprint,
    TurnLeft,
    TurnRight,
    Decelerate,
    Jump,
    Slide,
    Tackle,
    Shield,
    Celebrate,
    Goalkeeper
};

// Ball interaction states
UENUM(BlueprintType)
enum class EHMBallState : uint8
{
    NoBall,
    Receiving,
    Dribbling,
    Passing,
    Shooting,
    Crossing,
    Heading,
    Trapping,
    Shielding
};

// Per-player animation style overrides
USTRUCT(BlueprintType)
struct FPlayerAnimStyle
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float StrideLengthScale = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float ArmSwingIntensity = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float SprintLeanAngle = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float HipRotationScale = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float KneeLiftScale = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float CadenceScale = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float DecelerationSharpness = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    float TurnLeadBody = 0.0f;
};

UCLASS(ClassGroup=(HyperMotion), meta=(BlueprintSpawnableComponent))
class HYPERMOTION_API UHyperMotionComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UHyperMotionComponent();

    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;

    // --- State ---

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|State")
    EHMMovementState MovementState = EHMMovementState::Idle;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|State")
    EHMBallState BallState = EHMBallState::NoBall;

    // --- Computed values ---

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float Speed = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float NormalizedSpeed = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float Direction = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float LeanAngle = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float Fatigue = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float TransitionAlpha = 0.0f;

    // --- Configuration ---

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    FPlayerAnimStyle PlayerStyle;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    FString PlayerID;

    // Speed thresholds (cm/s)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    float IdleThreshold = 10.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    float WalkThreshold = 120.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    float JogThreshold = 250.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    float RunThreshold = 450.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "HyperMotion|Config")
    float MaxSpeed = 800.0f;

    // --- Functions ---

    UFUNCTION(BlueprintCallable, Category = "HyperMotion")
    void SetBallState(EHMBallState NewState);

    UFUNCTION(BlueprintCallable, Category = "HyperMotion")
    void SetFatigue(float NewFatigue);

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    bool IsMoving() const;

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    bool IsSprinting() const;

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    bool IsTurning() const;

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    bool HasBall() const;

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    float GetAcceleration() const;

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    float GetTurnRate() const;

    UFUNCTION(BlueprintCallable, Category = "HyperMotion")
    void ForceMovementState(EHMMovementState NewState, float Duration = 0.5f);

    UFUNCTION(BlueprintPure, Category = "HyperMotion")
    float GetTimeInCurrentState() const;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float SmoothedSpeed = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float Acceleration = 0.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    bool bIsAccelerating = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    bool bIsDecelerating = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "HyperMotion|Computed")
    float StrideFrequency = 1.0f;

private:
    void UpdateMovementState();
    void ComputeDerivedValues(float DeltaTime);
    void UpdateFatigue(float DeltaTime);
    EHMMovementState ClassifySpeedState(float CurrentSpeed) const;

    FVector PreviousPosition;
    FVector CurrentVelocity;
    FVector PreviousVelocity;
    float PreviousDirection = 0.0f;
    float TurnRate = 0.0f;
    float PreviousSpeed = 0.0f;
    float TimeInState = 0.0f;
    float ForcedStateDuration = 0.0f;
    bool bHasForcedState = false;
    EHMMovementState ForcedState = EHMMovementState::Idle;

    static constexpr float SpeedSmoothFactor = 0.15f;
    static constexpr float FatigueSprintRate = 0.02f;
    static constexpr float FatigueRecoveryRate = 0.03f;
};
