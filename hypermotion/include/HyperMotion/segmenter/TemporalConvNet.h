#pragma once

#include "HyperMotion/core/Types.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

namespace hm::segmenter {

// Temporal Convolutional Network for motion classification
// 6 dilated causal conv blocks, dilation [1,2,4,8,16,32], receptive field ~190 frames
// Hidden: 128 channels, kernel: 3
// Each block: Conv1D -> BN -> ReLU -> Dropout(0.1) -> Conv1D -> BN -> ReLU + residual
// Output: Conv1D(128, 16, k=1) -> per-frame logits
// ~600K parameters

struct TCNBlockImpl : torch::nn::Module {
    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Conv1d residual_conv{nullptr};

    TCNBlockImpl(int inChannels, int outChannels, int kernelSize, int dilation, float dropoutRate);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TCNBlock);

struct TemporalConvNetImpl : torch::nn::Module {
    std::vector<TCNBlock> blocks;
    torch::nn::Conv1d input_proj{nullptr};
    torch::nn::Conv1d output_proj{nullptr};

    TemporalConvNetImpl(int inputDim = 70, int hiddenDim = 128,
                         int numClasses = MOTION_TYPE_COUNT, int kernelSize = 3,
                         float dropout = 0.1f);
    torch::Tensor forward(torch::Tensor x);  // [batch, features, time] -> [batch, classes, time]
};
TORCH_MODULE(TemporalConvNet);

} // namespace hm::segmenter
