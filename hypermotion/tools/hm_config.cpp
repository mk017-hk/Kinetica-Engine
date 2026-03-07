#include "HyperMotion/core/PipelineConfigIO.h"
#include <iostream>
#include <string>

static void printUsage() {
    std::cout << "Usage: hm_config [options]\n"
              << "  Generates a template HyperMotion pipeline config file.\n\n"
              << "Options:\n"
              << "  --output <path>  Output file (default: hm_config.json)\n"
              << "  --print          Print to stdout instead of writing a file\n"
              << "  --help           Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string outputPath = "hm_config.json";
    bool printOnly = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--print") printOnly = true;
        else if (arg == "--help") { printUsage(); return 0; }
    }

    hm::PipelineConfig defaults;
    std::string json = hm::serialisePipelineConfig(defaults);

    if (printOnly) {
        std::cout << json << "\n";
    } else {
        if (hm::savePipelineConfig(outputPath, defaults)) {
            std::cout << "Template config written to: " << outputPath << "\n";
        } else {
            std::cerr << "Failed to write config file\n";
            return 1;
        }
    }
    return 0;
}
