#include "HyperMotionModule.h"

#define LOCTEXT_NAMESPACE "FHyperMotionModule"

void FHyperMotionModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("HyperMotion module started"));
}

void FHyperMotionModule::ShutdownModule()
{
    UE_LOG(LogTemp, Log, TEXT("HyperMotion module shut down"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FHyperMotionModule, HyperMotion)
