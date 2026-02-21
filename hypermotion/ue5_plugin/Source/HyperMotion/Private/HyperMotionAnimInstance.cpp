#include "HyperMotionAnimInstance.h"
#include "HyperMotionComponent.h"
#include "GameFramework/Character.h"

UHyperMotionAnimInstance::UHyperMotionAnimInstance()
{
}

void UHyperMotionAnimInstance::NativeInitializeAnimation()
{
    Super::NativeInitializeAnimation();

    APawn* OwningPawn = TryGetPawnOwner();
    if (OwningPawn)
    {
        HyperMotionComp = OwningPawn->FindComponentByClass<UHyperMotionComponent>();
    }
}

void UHyperMotionAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
    Super::NativeUpdateAnimation(DeltaSeconds);

    if (!HyperMotionComp.IsValid())
    {
        APawn* OwningPawn = TryGetPawnOwner();
        if (OwningPawn)
        {
            HyperMotionComp = OwningPawn->FindComponentByClass<UHyperMotionComponent>();
        }
    }

    if (HyperMotionComp.IsValid())
    {
        UHyperMotionComponent* Comp = HyperMotionComp.Get();

        Speed = Comp->Speed;
        NormalizedSpeed = Comp->NormalizedSpeed;
        Direction = Comp->Direction;
        LeanAngle = Comp->LeanAngle;
        Fatigue = Comp->Fatigue;
        TransitionAlpha = Comp->TransitionAlpha;

        bIsMoving = Comp->IsMoving();
        bIsSprinting = Comp->IsSprinting();
        bIsTurning = Comp->IsTurning();
        bHasBall = Comp->HasBall();

        MovementState = Comp->MovementState;
        BallState = Comp->BallState;
        PlayerStyle = Comp->PlayerStyle;
    }
}
