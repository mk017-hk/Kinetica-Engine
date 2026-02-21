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
}

void UHyperMotionComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                            FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    ComputeDerivedValues(DeltaTime);
    UpdateMovementState();
}

void UHyperMotionComponent::ComputeDerivedValues(float DeltaTime)
{
    if (DeltaTime <= 0.0f) return;

    FVector CurrentPosition = GetOwner()->GetActorLocation();
    CurrentVelocity = (CurrentPosition - PreviousPosition) / DeltaTime;
    PreviousPosition = CurrentPosition;

    // Speed (horizontal only, cm/s)
    FVector HorizontalVelocity = FVector(CurrentVelocity.X, CurrentVelocity.Y, 0.0f);
    Speed = HorizontalVelocity.Size();

    // Normalized speed [0, 1]
    NormalizedSpeed = FMath::Clamp(Speed / MaxSpeed, 0.0f, 1.0f);

    // Direction relative to actor forward (-180 to 180)
    if (Speed > IdleThreshold)
    {
        FVector Forward = GetOwner()->GetActorForwardVector();
        FVector VelDir = HorizontalVelocity.GetSafeNormal();
        float DotForward = FVector::DotProduct(Forward, VelDir);
        float DotRight = FVector::DotProduct(GetOwner()->GetActorRightVector(), VelDir);
        Direction = FMath::Atan2(DotRight, DotForward) * (180.0f / PI);
    }

    // Turn rate
    float DirectionDelta = Direction - PreviousDirection;
    if (DirectionDelta > 180.0f) DirectionDelta -= 360.0f;
    if (DirectionDelta < -180.0f) DirectionDelta += 360.0f;
    TurnRate = DirectionDelta / DeltaTime;
    PreviousDirection = Direction;

    // Lean angle based on turn rate and speed
    float SpeedFactor = FMath::Clamp(Speed / RunThreshold, 0.0f, 1.0f);
    LeanAngle = FMath::Clamp(TurnRate * 0.05f * SpeedFactor,
                              -PlayerStyle.SprintLeanAngle - 15.0f,
                               PlayerStyle.SprintLeanAngle + 15.0f);

    // Transition alpha (smoothed state blending)
    TransitionAlpha = FMath::FInterpTo(TransitionAlpha, 1.0f, DeltaTime, 8.0f);
}

void UHyperMotionComponent::UpdateMovementState()
{
    EHMMovementState PrevState = MovementState;

    // Check for special states first
    if (FMath::Abs(TurnRate) > 90.0f && Speed > IdleThreshold)
    {
        MovementState = (TurnRate > 0) ? EHMMovementState::TurnRight : EHMMovementState::TurnLeft;
    }
    // Speed-based states
    else if (Speed < IdleThreshold)
    {
        MovementState = EHMMovementState::Idle;
    }
    else if (Speed < WalkThreshold)
    {
        MovementState = EHMMovementState::Walk;
    }
    else if (Speed < JogThreshold)
    {
        MovementState = EHMMovementState::Jog;
    }
    else if (Speed < RunThreshold)
    {
        MovementState = EHMMovementState::Run;
    }
    else
    {
        MovementState = EHMMovementState::Sprint;
    }

    // Deceleration detection
    if (PrevState == EHMMovementState::Sprint || PrevState == EHMMovementState::Run)
    {
        if (MovementState == EHMMovementState::Jog || MovementState == EHMMovementState::Walk)
        {
            MovementState = EHMMovementState::Decelerate;
        }
    }

    // Reset transition alpha on state change
    if (MovementState != PrevState)
    {
        TransitionAlpha = 0.0f;
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
    return Speed > IdleThreshold;
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
