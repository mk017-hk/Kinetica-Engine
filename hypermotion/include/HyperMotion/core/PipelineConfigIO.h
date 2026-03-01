#pragma once

#include "HyperMotion/Pipeline.h"
#include <string>

namespace hm {

/// Load a PipelineConfig from a JSON file.  Returns true on success.
/// Fields not present in the file keep their default values.
bool loadPipelineConfig(const std::string& path, PipelineConfig& out);

/// Load a PipelineConfig from a JSON string.
bool parsePipelineConfig(const std::string& json, PipelineConfig& out);

/// Serialise a PipelineConfig to a pretty-printed JSON string.
std::string serialisePipelineConfig(const PipelineConfig& config);

/// Write a PipelineConfig to a JSON file.  Returns true on success.
bool savePipelineConfig(const std::string& path, const PipelineConfig& config);

/// Serialise a PipelineStats to a JSON string.
std::string serialisePipelineStats(const PipelineStats& stats);

/// Write a PipelineStats to a JSON file.  Returns true on success.
bool savePipelineStats(const std::string& path, const PipelineStats& stats);

} // namespace hm
