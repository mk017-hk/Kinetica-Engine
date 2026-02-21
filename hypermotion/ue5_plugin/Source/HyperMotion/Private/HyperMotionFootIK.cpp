#include "HyperMotionFootIK.h"
#include "Components/SkeletalMeshComponent.h"
#include "DrawDebugHelpers.h"

UHyperMotionFootIK::UHyperMotionFootIK()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UHyperMotionFootIK::BeginPlay()
{
    Super::BeginPlay();

    MeshComp = GetOwner()->FindComponentByClass<USkeletalMeshComponent>();
    if (!MeshComp)
    {
        UE_LOG(LogTemp, Warning, TEXT("HyperMotionFootIK: No SkeletalMeshComponent found"));
    }
}

void UHyperMotionFootIK::TickComponent(float DeltaTime, ELevelTick TickType,
                                         FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!MeshComp) return;

    // Get foot bone world positions
    FVector LeftFootPos = MeshComp->GetSocketLocation(LeftFootBone);
    FVector RightFootPos = MeshComp->GetSocketLocation(RightFootBone);

    // Trace from each foot
    FFootTraceResult LeftTrace = TraceFootPosition(LeftFootPos);
    FFootTraceResult RightTrace = TraceFootPosition(RightFootPos);

    // Compute target offsets
    FVector TargetLeftOffset = FVector::ZeroVector;
    FVector TargetRightOffset = FVector::ZeroVector;
    float TargetHipOffset = 0.0f;

    FootIKData.bLeftFootGrounded = LeftTrace.bHit;
    FootIKData.bRightFootGrounded = RightTrace.bHit;

    if (LeftTrace.bHit)
    {
        float LeftDelta = LeftTrace.ImpactPoint.Z - LeftFootPos.Z + FootOffset;
        TargetLeftOffset = FVector(0.0f, 0.0f, LeftDelta);
        FootIKData.LeftFootRotation = ComputeFootRotation(LeftTrace.ImpactNormal);
    }

    if (RightTrace.bHit)
    {
        float RightDelta = RightTrace.ImpactPoint.Z - RightFootPos.Z + FootOffset;
        TargetRightOffset = FVector(0.0f, 0.0f, RightDelta);
        FootIKData.RightFootRotation = ComputeFootRotation(RightTrace.ImpactNormal);
    }

    // Hip offset: lower the hips by the lowest foot offset to keep feet on ground
    if (LeftTrace.bHit && RightTrace.bHit)
    {
        TargetHipOffset = FMath::Min(TargetLeftOffset.Z, TargetRightOffset.Z);
    }
    else if (LeftTrace.bHit)
    {
        TargetHipOffset = TargetLeftOffset.Z;
    }
    else if (RightTrace.bHit)
    {
        TargetHipOffset = TargetRightOffset.Z;
    }

    // Smooth interpolation
    CurrentLeftFootOffset = FMath::VInterpTo(CurrentLeftFootOffset, TargetLeftOffset,
                                              DeltaTime, InterpSpeed);
    CurrentRightFootOffset = FMath::VInterpTo(CurrentRightFootOffset, TargetRightOffset,
                                               DeltaTime, InterpSpeed);
    CurrentHipOffset = FMath::FInterpTo(CurrentHipOffset, TargetHipOffset,
                                         DeltaTime, InterpSpeed);

    // Apply relative to hip offset
    FootIKData.LeftFootOffset = CurrentLeftFootOffset - FVector(0, 0, CurrentHipOffset);
    FootIKData.RightFootOffset = CurrentRightFootOffset - FVector(0, 0, CurrentHipOffset);
    FootIKData.HipOffset = CurrentHipOffset;
}

UHyperMotionFootIK::FFootTraceResult UHyperMotionFootIK::TraceFootPosition(
    const FVector& FootWorldPos)
{
    FFootTraceResult Result;

    FVector TraceStart = FootWorldPos + FVector(0.0f, 0.0f, TraceDistance * 0.5f);
    FVector TraceEnd = FootWorldPos - FVector(0.0f, 0.0f, TraceDistance);

    FHitResult HitResult;
    FCollisionQueryParams QueryParams;
    QueryParams.AddIgnoredActor(GetOwner());
    QueryParams.bTraceComplex = false;

    bool bHit = GetWorld()->LineTraceSingleByChannel(
        HitResult, TraceStart, TraceEnd, TraceChannel, QueryParams);

    if (bHit)
    {
        Result.bHit = true;
        Result.ImpactPoint = HitResult.ImpactPoint;
        Result.ImpactNormal = HitResult.ImpactNormal;
        Result.Distance = HitResult.Distance;
    }

    return Result;
}

FRotator UHyperMotionFootIK::ComputeFootRotation(const FVector& ImpactNormal)
{
    // Compute rotation to align foot with ground normal
    FVector UpVector = FVector::UpVector;
    FQuat RotationQuat = FQuat::FindBetweenNormals(UpVector, ImpactNormal);
    return RotationQuat.Rotator();
}
