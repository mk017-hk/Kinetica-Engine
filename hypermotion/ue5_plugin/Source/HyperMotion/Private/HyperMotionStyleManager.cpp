#include "HyperMotionStyleManager.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

UHyperMotionStyleManager::UHyperMotionStyleManager()
{
}

void UHyperMotionStyleManager::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("HyperMotion Style Manager initialized"));
}

void UHyperMotionStyleManager::Deinitialize()
{
    Super::Deinitialize();
}

bool UHyperMotionStyleManager::LoadStyleLibrary(const FString& JsonPath)
{
    FString JsonString;
    if (!FFileHelper::LoadFileToString(JsonString, *JsonPath))
    {
        UE_LOG(LogTemp, Error, TEXT("HyperMotion: Cannot load style file: %s"), *JsonPath);
        return false;
    }

    TSharedPtr<FJsonObject> RootObj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
    if (!FJsonSerializer::Deserialize(Reader, RootObj) || !RootObj.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("HyperMotion: Failed to parse JSON: %s"), *JsonPath);
        return false;
    }

    StyleMap.Empty();

    const TArray<TSharedPtr<FJsonValue>>* StylesArray;
    if (RootObj->TryGetArrayField(TEXT("styles"), StylesArray))
    {
        for (const auto& StyleValue : *StylesArray)
        {
            const TSharedPtr<FJsonObject>& StyleObj = StyleValue->AsObject();
            if (!StyleObj.IsValid()) continue;

            FHMStyleEntry Entry;
            Entry.PlayerID = StyleObj->GetStringField(TEXT("playerID"));
            Entry.PlayerName = StyleObj->GetStringField(TEXT("playerName"));

            // Load style parameters
            Entry.Style.StrideLengthScale = StyleObj->GetNumberField(TEXT("strideLengthScale"));
            Entry.Style.ArmSwingIntensity = StyleObj->GetNumberField(TEXT("armSwingIntensity"));
            Entry.Style.SprintLeanAngle = StyleObj->GetNumberField(TEXT("sprintLeanAngle"));
            Entry.Style.HipRotationScale = StyleObj->GetNumberField(TEXT("hipRotationScale"));
            Entry.Style.KneeLiftScale = StyleObj->GetNumberField(TEXT("kneeLiftScale"));
            Entry.Style.CadenceScale = StyleObj->GetNumberField(TEXT("cadenceScale"));
            Entry.Style.DecelerationSharpness = StyleObj->GetNumberField(TEXT("decelerationSharpness"));
            Entry.Style.TurnLeadBody = StyleObj->GetNumberField(TEXT("turnLeadBody"));

            // Load embedding
            const TArray<TSharedPtr<FJsonValue>>* EmbeddingArray;
            if (StyleObj->TryGetArrayField(TEXT("embedding"), EmbeddingArray))
            {
                Entry.Embedding.Reserve(EmbeddingArray->Num());
                for (const auto& Val : *EmbeddingArray)
                {
                    Entry.Embedding.Add(static_cast<float>(Val->AsNumber()));
                }
            }

            StyleMap.Add(Entry.PlayerID, Entry);
        }
    }

    UE_LOG(LogTemp, Log, TEXT("HyperMotion: Loaded %d player styles from %s"),
           StyleMap.Num(), *JsonPath);
    return true;
}

bool UHyperMotionStyleManager::SaveStyleLibrary(const FString& JsonPath)
{
    TSharedPtr<FJsonObject> RootObj = MakeShareable(new FJsonObject);
    RootObj->SetStringField(TEXT("version"), TEXT("1.0"));

    TArray<TSharedPtr<FJsonValue>> StylesArray;
    for (const auto& Pair : StyleMap)
    {
        const FHMStyleEntry& Entry = Pair.Value;
        TSharedPtr<FJsonObject> StyleObj = MakeShareable(new FJsonObject);

        StyleObj->SetStringField(TEXT("playerID"), Entry.PlayerID);
        StyleObj->SetStringField(TEXT("playerName"), Entry.PlayerName);
        StyleObj->SetNumberField(TEXT("strideLengthScale"), Entry.Style.StrideLengthScale);
        StyleObj->SetNumberField(TEXT("armSwingIntensity"), Entry.Style.ArmSwingIntensity);
        StyleObj->SetNumberField(TEXT("sprintLeanAngle"), Entry.Style.SprintLeanAngle);
        StyleObj->SetNumberField(TEXT("hipRotationScale"), Entry.Style.HipRotationScale);
        StyleObj->SetNumberField(TEXT("kneeLiftScale"), Entry.Style.KneeLiftScale);
        StyleObj->SetNumberField(TEXT("cadenceScale"), Entry.Style.CadenceScale);
        StyleObj->SetNumberField(TEXT("decelerationSharpness"), Entry.Style.DecelerationSharpness);
        StyleObj->SetNumberField(TEXT("turnLeadBody"), Entry.Style.TurnLeadBody);

        TArray<TSharedPtr<FJsonValue>> EmbeddingValues;
        for (float V : Entry.Embedding)
        {
            EmbeddingValues.Add(MakeShareable(new FJsonValueNumber(V)));
        }
        StyleObj->SetArrayField(TEXT("embedding"), EmbeddingValues);

        StylesArray.Add(MakeShareable(new FJsonValueObject(StyleObj)));
    }
    RootObj->SetArrayField(TEXT("styles"), StylesArray);

    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    FJsonSerializer::Serialize(RootObj.ToSharedRef(), Writer);

    return FFileHelper::SaveStringToFile(OutputString, *JsonPath);
}

bool UHyperMotionStyleManager::GetPlayerStyle(const FString& PlayerID, FPlayerAnimStyle& OutStyle)
{
    const FHMStyleEntry* Entry = StyleMap.Find(PlayerID);
    if (Entry)
    {
        OutStyle = Entry->Style;
        return true;
    }
    return false;
}

void UHyperMotionStyleManager::SetPlayerStyle(const FString& PlayerID,
                                                const FPlayerAnimStyle& Style)
{
    FHMStyleEntry& Entry = StyleMap.FindOrAdd(PlayerID);
    Entry.PlayerID = PlayerID;
    Entry.Style = Style;
}

FPlayerAnimStyle UHyperMotionStyleManager::InterpolateStyles(
    const FPlayerAnimStyle& StyleA, const FPlayerAnimStyle& StyleB, float Alpha)
{
    FPlayerAnimStyle Result;
    Alpha = FMath::Clamp(Alpha, 0.0f, 1.0f);

    Result.StrideLengthScale = FMath::Lerp(StyleA.StrideLengthScale, StyleB.StrideLengthScale, Alpha);
    Result.ArmSwingIntensity = FMath::Lerp(StyleA.ArmSwingIntensity, StyleB.ArmSwingIntensity, Alpha);
    Result.SprintLeanAngle = FMath::Lerp(StyleA.SprintLeanAngle, StyleB.SprintLeanAngle, Alpha);
    Result.HipRotationScale = FMath::Lerp(StyleA.HipRotationScale, StyleB.HipRotationScale, Alpha);
    Result.KneeLiftScale = FMath::Lerp(StyleA.KneeLiftScale, StyleB.KneeLiftScale, Alpha);
    Result.CadenceScale = FMath::Lerp(StyleA.CadenceScale, StyleB.CadenceScale, Alpha);
    Result.DecelerationSharpness = FMath::Lerp(StyleA.DecelerationSharpness, StyleB.DecelerationSharpness, Alpha);
    Result.TurnLeadBody = FMath::Lerp(StyleA.TurnLeadBody, StyleB.TurnLeadBody, Alpha);

    return Result;
}

void UHyperMotionStyleManager::ApplyStyleToComponent(
    UHyperMotionComponent* Component, const FString& PlayerID)
{
    if (!Component) return;

    FPlayerAnimStyle Style;
    if (GetPlayerStyle(PlayerID, Style))
    {
        Component->PlayerStyle = Style;
        Component->PlayerID = PlayerID;
    }
}

TArray<FString> UHyperMotionStyleManager::GetAllPlayerIDs() const
{
    TArray<FString> IDs;
    StyleMap.GetKeys(IDs);
    return IDs;
}

int32 UHyperMotionStyleManager::GetStyleCount() const
{
    return StyleMap.Num();
}
