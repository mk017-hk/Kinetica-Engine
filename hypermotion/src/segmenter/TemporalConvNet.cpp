#include "HyperMotion/segmenter/TemporalConvNet.h"

namespace hm::segmenter {

// -------------------------------------------------------------------
// TCN Block
// -------------------------------------------------------------------

TCNBlockImpl::TCNBlockImpl(int inChannels, int outChannels, int kernelSize,
                             int dilation, float dropoutRate) {
    int padding = (kernelSize - 1) * dilation;

    conv1 = register_module("conv1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize)
            .dilation(dilation).padding(padding)));
    bn1 = register_module("bn1", torch::nn::BatchNorm1d(outChannels));

    conv2 = register_module("conv2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(outChannels, outChannels, kernelSize)
            .dilation(dilation).padding(padding)));
    bn2 = register_module("bn2", torch::nn::BatchNorm1d(outChannels));

    dropout = register_module("dropout", torch::nn::Dropout(dropoutRate));

    if (inChannels != outChannels) {
        residual_conv = register_module("residual_conv",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, 1)));
    }
}

torch::Tensor TCNBlockImpl::forward(torch::Tensor x) {
    auto residual = x;

    // Conv1 -> BN -> ReLU -> Dropout
    auto out = conv1(x);
    // Causal: trim future (right side) to maintain causal behavior
    if (out.size(2) > x.size(2)) {
        out = out.slice(2, 0, x.size(2));
    }
    out = torch::relu(bn1(out));
    out = dropout(out);

    // Conv2 -> BN -> ReLU
    out = conv2(out);
    if (out.size(2) > x.size(2)) {
        out = out.slice(2, 0, x.size(2));
    }
    out = bn2(out);

    // Residual connection
    if (residual_conv.is_empty()) {
        out = torch::relu(out + residual);
    } else {
        out = torch::relu(out + residual_conv(residual));
    }

    return out;
}

// -------------------------------------------------------------------
// Temporal Convolutional Network
// -------------------------------------------------------------------

TemporalConvNetImpl::TemporalConvNetImpl(int inputDim, int hiddenDim,
                                           int numClasses, int kernelSize,
                                           float dropout) {
    // Input projection
    input_proj = register_module("input_proj",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(inputDim, hiddenDim, 1)));

    // 6 dilated causal conv blocks: dilation = [1, 2, 4, 8, 16, 32]
    std::vector<int> dilations = {1, 2, 4, 8, 16, 32};
    for (size_t i = 0; i < dilations.size(); ++i) {
        auto block = TCNBlock(hiddenDim, hiddenDim, kernelSize, dilations[i], dropout);
        blocks.push_back(register_module("block_" + std::to_string(i), block));
    }

    // Output projection: per-frame classification
    output_proj = register_module("output_proj",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(hiddenDim, numClasses, 1)));
}

torch::Tensor TemporalConvNetImpl::forward(torch::Tensor x) {
    // x: [batch, features, time]
    x = input_proj(x);

    for (auto& block : blocks) {
        x = block(x);
    }

    x = output_proj(x);
    return x; // [batch, numClasses, time]
}

} // namespace hm::segmenter
