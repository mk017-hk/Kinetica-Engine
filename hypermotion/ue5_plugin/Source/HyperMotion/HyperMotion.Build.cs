using UnrealBuildTool;
using System.IO;

public class HyperMotion : ModuleRules
{
    public HyperMotion(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicIncludePaths.AddRange(new string[] {
            Path.Combine(ModuleDirectory, "Public")
        });

        PrivateIncludePaths.AddRange(new string[] {
            Path.Combine(ModuleDirectory, "Private")
        });

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "AnimGraphRuntime"
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "RenderCore",
            "RHI",
            "Projects",
            "InputCore"
        });

        // ONNX Runtime for ML inference
        string OnnxRuntimePath = Path.Combine(ModuleDirectory, "..", "..", "ThirdParty", "OnnxRuntime");
        if (Directory.Exists(OnnxRuntimePath))
        {
            PublicIncludePaths.Add(Path.Combine(OnnxRuntimePath, "include"));
            PublicAdditionalLibraries.Add(Path.Combine(OnnxRuntimePath, "lib", "onnxruntime.lib"));
            RuntimeDependencies.Add(Path.Combine(OnnxRuntimePath, "lib", "onnxruntime.dll"));
            PublicDefinitions.Add("HM_UE5_ONNX=1");
        }

        PublicDefinitions.Add("HM_UE5_PLUGIN=1");
    }
}
