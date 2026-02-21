#pragma once

#include "HyperMotion/core/Types.h"
#include <torch/torch.h>

namespace hm::style {

// Style Encoder: Variable-length motion -> 64D style embedding
// Input: [batch, time, 201] (132 rotations + 3 root vel + 66 angular vel)
// Conv1D(201,128) -> 4 ResBlocks (128->128->256->256->512) -> GAP -> Linear(512,256)
//   -> ReLU -> Linear(256,64) -> L2 norm
// ~1.9M parameters

static constexpr int STYLE_INPUT_DIM = 201;  // 132 + 3 + 66

struct StyleResBlockImpl : torch::nn::Module {
    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr};
    torch::nn::Conv1d downsample{nullptr};

    StyleResBlockImpl(int inChannels, int outChannels);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(StyleResBlock);

struct StyleEncoderImpl : torch::nn::Module {
    torch::nn::Conv1d input_conv{nullptr};
    torch::nn::BatchNorm1d input_bn{nullptr};
    StyleResBlock res1{nullptr}, res2{nullptr}, res3{nullptr}, res4{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    StyleEncoderImpl();

    // Input: [batch, time, 201]
    // Output: [batch, 64] (L2 normalized)
    torch::Tensor forward(torch::Tensor x);

    // Prepare input from skeleton frames
    static torch::Tensor prepareInput(const std::vector<SkeletonFrame>& frames);
};
TORCH_MODULE(StyleEncoder);

} // namespace hm::style
