#include "HyperMotionComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"

UHyperMotionComponent::UHyperMotionComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UHyperMotionComponent::BeginPlay()
{
    Super::BeginPlay();
    PreviousPosition = GetOwner()->GetActorLocation();
    PreviousVelocity = FVector::ZeroVector;
}

void UHyperMotionComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                            FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    ComputeDerivedValues(DeltaTime);
    UpdateFatigue(DeltaTime);
    UpdateMovementState();

    TimeInState += DeltaTime;

    // Handle forced state duration
    if (bHasForcedState)
    {
        ForcedStateDuration -= DeltaTime;
        if (ForcedStateDuration <= 0.0f)
        {
            bHasForcedState = false;
        }
    }
}

void UHyperMotionComponent::ComputeDerivedValues(float DeltaTime)
{
    if (DeltaTime <= 0.0f) return;

    FVector CurrentPosition = GetOwner()->GetActorLocation();
    PreviousVelocity = CurrentVelocity;
    CurrentVelocity = (CurrentPosition - PreviousPosition) / DeltaTime;
    PreviousPosition = CurrentPosition;

    // Speed (horizontal only, cm/s)
    FVector HorizontalVelocity = FVector(CurrentVelocity.X, CurrentVelocity.Y, 0.0f);
    PreviousSpeed = Speed;
    Speed = HorizontalVelocity.Size();

    // Smoothed speed (EMA filter for animation stability)
    SmoothedSpeed = FMath::Lerp(SmoothedSpeed, Speed, SpeedSmoothFactor);

    // Normalized speed [0, 1]
    NormalizedSpeed = FMath::Clamp(SmoothedSpeed / MaxSpeed, 0.0f, 1.0f);

    // Acceleration (cm/s^2)
    float SpeedDelta = Speed - PreviousSpeed;
    Acceleration = SpeedDelta / DeltaTime;
    bIsAccelerating = Acceleration > 50.0f;
    bIsDecelerating = Acceleration < -100.0f;

    // Direction relative to actor forward (-180 to 180)
    if (Speed > IdleThreshold)
    {
        FVector Forward = GetOwner()->GetActorForwardVector();
        FVector VelDir = HorizontalVelocity.GetSafeNormal();
        float DotForward = FVector::DotProduct(Forward, VelDir);
        float DotRight = FVector::DotProduct(GetOwner()->GetActorRightVector(), VelDir);
        Direction = FMath::Atan2(DotRight, DotForward) * (180.0f / PI);
    }

    // Turn rate (deg/s)
    float DirectionDelta = Direction - PreviousDirection;
    if (DirectionDelta > 180.0f) DirectionDelta -= 360.0f;
    if (DirectionDelta < -180.0f) DirectionDelta += 360.0f;
    TurnRate = DirectionDelta / DeltaTime;
    PreviousDirection = Direction;

    // Lean angle based on turn rate and speed
    float SpeedFactor = FMath::Clamp(SmoothedSpeed / RunThreshold, 0.0f, 1.0f);
    float BaseLean = TurnRate * 0.05f * SpeedFactor;
    float MaxLean = PlayerStyle.SprintLeanAngle + 15.0f;
    LeanAngle = FMath::Clamp(BaseLean, -MaxLean, MaxLean);

    // Stride frequency: increases with speed, modulated by player cadence style
    float SpeedRatio = FMath::Clamp(SmoothedSpeed / RunThreshold, 0.0f, 2.0f);
    StrideFrequency = FMath::Lerp(0.5f, 2.0f, SpeedRatio) * PlayerStyle.CadenceScale;

    // Transition alpha (smoothed state blending)
    TransitionAlpha = FMath::FInterpTo(TransitionAlpha, 1.0f, DeltaTime, 8.0f);
}

void UHyperMotionComponent::UpdateFatigue(float DeltaTime)
{
    // Fatigue increases with sprinting, recovers at lower speeds
    if (Speed > RunThreshold)
    {
        float SprintIntensity = FMath::Clamp(
            (Speed - RunThreshold) / (MaxSpeed - RunThreshold), 0.0f, 1.0f);
        Fatigue = FMath::Clamp(
            Fatigue + FatigueSprintRate * SprintIntensity * DeltaTime, 0.0f, 1.0f);
    }
    else if (Speed < WalkThreshold)
    {
        float RecoveryMultiplier = (Speed < IdleThreshold) ? 1.5f : 1.0f;
        Fatigue = FMath::Clamp(
            Fatigue - FatigueRecoveryRate * RecoveryMultiplier * DeltaTime, 0.0f, 1.0f);
    }
    else
    {
        // Moderate speeds: slow recovery
        Fatigue = FMath::Clamp(
            Fatigue - FatigueRecoveryRate * 0.3f * DeltaTime, 0.0f, 1.0f);
    }
}

EHMMovementState UHyperMotionComponent::ClassifySpeedState(float CurrentSpeed) const
{
    if (CurrentSpeed < IdleThreshold) return EHMMovementState::Idle;
    if (CurrentSpeed < WalkThreshold) return EHMMovementState::Walk;
    if (CurrentSpeed < JogThreshold)  return EHMMovementState::Jog;
    if (CurrentSpeed < RunThreshold)  return EHMMovementState::Run;
    return EHMMovementState::Sprint;
}

void UHyperMotionComponent::UpdateMovementState()
{
    // Forced state override
    if (bHasForcedState)
    {
        if (MovementState != ForcedState)
        {
            MovementState = ForcedState;
            TransitionAlpha = 0.0f;
            TimeInState = 0.0f;
        }
        return;
    }

    EHMMovementState PrevState = MovementState;

    // Check for special states first: sharp turns
    if (FMath::Abs(TurnRate) > 90.0f && Speed > IdleThreshold)
    {
        MovementState = (TurnRate > 0)
            ? EHMMovementState::TurnRight
            : EHMMovementState::TurnLeft;
    }
    else
    {
        // Speed-based states (use smoothed speed for stability)
        MovementState = ClassifySpeedState(SmoothedSpeed);
    }

    // Deceleration detection: rapid speed drop from high speed
    if (PrevState == EHMMovementState::Sprint || PrevState == EHMMovementState::Run)
    {
        if (bIsDecelerating && Acceleration < -200.0f)
        {
            MovementState = EHMMovementState::Decelerate;
        }
        else if (MovementState == EHMMovementState::Jog ||
                 MovementState == EHMMovementState::Walk)
        {
            // Gradual slow-down triggers decelerate briefly
            if (TimeInState < 0.3f)
            {
                MovementState = EHMMovementState::Decelerate;
            }
        }
    }

    // Decelerate state duration scaled by style sharpness
    if (MovementState == EHMMovementState::Decelerate &&
        TimeInState > 0.5f * (2.0f - FMath::Clamp(PlayerStyle.DecelerationSharpness, 0.0f, 2.0f)))
    {
        MovementState = ClassifySpeedState(SmoothedSpeed);
    }

    // Reset transition alpha on state change
    if (MovementState != PrevState)
    {
        TransitionAlpha = 0.0f;
        TimeInState = 0.0f;
    }
}

void UHyperMotionComponent::SetBallState(EHMBallState NewState)
{
    BallState = NewState;
}

void UHyperMotionComponent::SetFatigue(float NewFatigue)
{
    Fatigue = FMath::Clamp(NewFatigue, 0.0f, 1.0f);
}

bool UHyperMotionComponent::IsMoving() const
{
    return SmoothedSpeed > IdleThreshold;
}

bool UHyperMotionComponent::IsSprinting() const
{
    return MovementState == EHMMovementState::Sprint;
}

bool UHyperMotionComponent::IsTurning() const
{
    return MovementState == EHMMovementState::TurnLeft ||
           MovementState == EHMMovementState::TurnRight;
}

bool UHyperMotionComponent::HasBall() const
{
    return BallState != EHMBallState::NoBall;
}

float UHyperMotionComponent::GetAcceleration() const
{
    return Acceleration;
}

float UHyperMotionComponent::GetTurnRate() const
{
    return TurnRate;
}

void UHyperMotionComponent::ForceMovementState(EHMMovementState NewState, float Duration)
{
    bHasForcedState = true;
    ForcedState = NewState;
    ForcedStateDuration = Duration;
}

float UHyperMotionComponent::GetTimeInCurrentState() const
{
    return TimeInState;
}
