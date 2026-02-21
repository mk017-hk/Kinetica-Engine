#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "HyperMotionComponent.h"
#include "HyperMotionStyleManager.generated.h"

USTRUCT(BlueprintType)
struct FHMStyleEntry
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    FString PlayerID;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    FString PlayerName;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    FPlayerAnimStyle Style;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Style")
    TArray<float> Embedding;
};

UCLASS()
class HYPERMOTION_API UHyperMotionStyleManager : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    UHyperMotionStyleManager();

    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    // Load style library from JSON file
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    bool LoadStyleLibrary(const FString& JsonPath);

    // Save style library to JSON file
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    bool SaveStyleLibrary(const FString& JsonPath);

    // Get style for a player
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    bool GetPlayerStyle(const FString& PlayerID, FPlayerAnimStyle& OutStyle);

    // Set style for a player
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    void SetPlayerStyle(const FString& PlayerID, const FPlayerAnimStyle& Style);

    // Interpolate between two styles
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    FPlayerAnimStyle InterpolateStyles(const FPlayerAnimStyle& StyleA,
                                        const FPlayerAnimStyle& StyleB,
                                        float Alpha);

    // Apply style to a HyperMotion component
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    void ApplyStyleToComponent(UHyperMotionComponent* Component, const FString& PlayerID);

    // Get all player IDs
    UFUNCTION(BlueprintCallable, Category = "HyperMotion|Style")
    TArray<FString> GetAllPlayerIDs() const;

    UFUNCTION(BlueprintPure, Category = "HyperMotion|Style")
    int32 GetStyleCount() const;

private:
    UPROPERTY()
    TMap<FString, FHMStyleEntry> StyleMap;
};
