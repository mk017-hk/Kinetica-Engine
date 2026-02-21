#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/core/MathUtils.h"

namespace hm::style {

// -------------------------------------------------------------------
// Style ResBlock
// -------------------------------------------------------------------

StyleResBlockImpl::StyleResBlockImpl(int inChannels, int outChannels) {
    conv1 = register_module("conv1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, 3).padding(1)));
    bn1 = register_module("bn1", torch::nn::BatchNorm1d(outChannels));
    conv2 = register_module("conv2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(outChannels, outChannels, 3).padding(1)));
    bn2 = register_module("bn2", torch::nn::BatchNorm1d(outChannels));

    if (inChannels != outChannels) {
        downsample = register_module("downsample",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(inChannels, outChannels, 1)));
    }
}

torch::Tensor StyleResBlockImpl::forward(torch::Tensor x) {
    auto residual = x;

    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));

    if (!downsample.is_empty()) {
        residual = downsample(residual);
    }

    out = torch::relu(out + residual);
    return out;
}

// -------------------------------------------------------------------
// Style Encoder
// -------------------------------------------------------------------

StyleEncoderImpl::StyleEncoderImpl() {
    input_conv = register_module("input_conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(STYLE_INPUT_DIM, 128, 3).padding(1)));
    input_bn = register_module("input_bn", torch::nn::BatchNorm1d(128));

    // 4 ResBlocks: 128->128->256->256->512
    res1 = register_module("res1", StyleResBlock(128, 128));
    res2 = register_module("res2", StyleResBlock(128, 256));
    res3 = register_module("res3", StyleResBlock(256, 256));
    res4 = register_module("res4", StyleResBlock(256, 512));

    fc1 = register_module("fc1", torch::nn::Linear(512, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, STYLE_DIM));
}

torch::Tensor StyleEncoderImpl::forward(torch::Tensor x) {
    // x: [batch, time, 201]
    // Transpose to [batch, channels, time] for Conv1D
    x = x.transpose(1, 2);

    x = torch::relu(input_bn(input_conv(x)));  // [batch, 128, time]

    x = res1(x);  // [batch, 128, time]
    x = res2(x);  // [batch, 256, time]
    x = res3(x);  // [batch, 256, time]
    x = res4(x);  // [batch, 512, time]

    // Global Average Pooling over time
    x = torch::adaptive_avg_pool1d(x, 1).squeeze(-1);  // [batch, 512]

    x = torch::relu(fc1(x));  // [batch, 256]
    x = fc2(x);               // [batch, 64]

    // L2 normalize
    x = torch::nn::functional::normalize(x,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    return x;
}

torch::Tensor StyleEncoderImpl::prepareInput(const std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    auto tensor = torch::zeros({1, numFrames, STYLE_INPUT_DIM});
    auto accessor = tensor.accessor<float, 3>();

    for (int f = 0; f < numFrames; ++f) {
        int idx = 0;

        // 132D rotations (22 joints x 6D)
        for (int j = 0; j < JOINT_COUNT; ++j) {
            for (int d = 0; d < ROTATION_DIM; ++d) {
                accessor[0][f][idx++] = frames[f].joints[j].rotation6D[d];
            }
        }

        // 3D root velocity
        accessor[0][f][idx++] = frames[f].rootVelocity.x / 800.0f;
        accessor[0][f][idx++] = frames[f].rootVelocity.y / 800.0f;
        accessor[0][f][idx++] = frames[f].rootVelocity.z / 800.0f;

        // 66D angular velocities (22 joints x 3 Euler angular velocity)
        // Estimated from frame differences
        if (f > 0) {
            for (int j = 0; j < JOINT_COUNT; ++j) {
                Vec3 prevEuler = frames[f - 1].joints[j].localEulerDeg;
                Vec3 currEuler = frames[f].joints[j].localEulerDeg;
                accessor[0][f][idx++] = (currEuler.x - prevEuler.x) / 360.0f;
                accessor[0][f][idx++] = (currEuler.y - prevEuler.y) / 360.0f;
                accessor[0][f][idx++] = (currEuler.z - prevEuler.z) / 360.0f;
            }
        } else {
            idx += 66;
        }
    }

    return tensor;
}

} // namespace hm::style
