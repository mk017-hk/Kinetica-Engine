#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hm::style {

static constexpr int STYLE_INPUT_DIM = 201;  // 132 + 3 + 66

/// ONNX-based style encoder: variable-length motion -> 64D embedding.
/// The model is trained in Python and exported to ONNX.
class StyleEncoder {
public:
    StyleEncoder();
    ~StyleEncoder();

    StyleEncoder(const StyleEncoder&) = delete;
    StyleEncoder& operator=(const StyleEncoder&) = delete;
    StyleEncoder(StyleEncoder&&) noexcept;
    StyleEncoder& operator=(StyleEncoder&&) noexcept;

    bool load(const std::string& onnxPath, bool useGPU = true);
    bool isLoaded() const;

    /// Encode a motion clip to a 64D style embedding (L2 normalized).
    std::array<float, STYLE_DIM> encode(const std::vector<SkeletonFrame>& frames);

    /// Prepare the 201D-per-frame feature matrix from skeleton frames.
    static std::vector<std::array<float, STYLE_INPUT_DIM>>
    prepareInput(const std::vector<SkeletonFrame>& frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::style
