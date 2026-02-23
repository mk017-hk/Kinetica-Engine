#include "HyperMotion/pose/DepthLifter.h"
#include "HyperMotion/core/Logger.h"

#ifdef HM_HAS_TORCH
#include <torch/torch.h>
#endif

#include <cmath>
#include <algorithm>

namespace hm::pose {

static constexpr const char* TAG = "DepthLifter";

#ifdef HM_HAS_TORCH
// Learned lifting network:
// Linear(34,1024) -> BN -> ReLU -> [Linear(1024,1024) -> BN -> ReLU + skip] x2 -> Linear(1024,51)
struct LiftingNetImpl : torch::nn::Module {
    torch::nn::Linear fc_in{nullptr};
    torch::nn::BatchNorm1d bn_in{nullptr};

    // Residual blocks
    torch::nn::Linear res1_fc1{nullptr}, res1_fc2{nullptr};
    torch::nn::BatchNorm1d res1_bn1{nullptr}, res1_bn2{nullptr};

    torch::nn::Linear res2_fc1{nullptr}, res2_fc2{nullptr};
    torch::nn::BatchNorm1d res2_bn1{nullptr}, res2_bn2{nullptr};

    torch::nn::Linear fc_out{nullptr};

    LiftingNetImpl() {
        fc_in = register_module("fc_in", torch::nn::Linear(34, 1024));
        bn_in = register_module("bn_in", torch::nn::BatchNorm1d(1024));

        res1_fc1 = register_module("res1_fc1", torch::nn::Linear(1024, 1024));
        res1_bn1 = register_module("res1_bn1", torch::nn::BatchNorm1d(1024));
        res1_fc2 = register_module("res1_fc2", torch::nn::Linear(1024, 1024));
        res1_bn2 = register_module("res1_bn2", torch::nn::BatchNorm1d(1024));

        res2_fc1 = register_module("res2_fc1", torch::nn::Linear(1024, 1024));
        res2_bn1 = register_module("res2_bn1", torch::nn::BatchNorm1d(1024));
        res2_fc2 = register_module("res2_fc2", torch::nn::Linear(1024, 1024));
        res2_bn2 = register_module("res2_bn2", torch::nn::BatchNorm1d(1024));

        fc_out = register_module("fc_out", torch::nn::Linear(1024, 51));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn_in(fc_in(x)));

        // Residual block 1
        auto res = x;
        x = torch::relu(res1_bn1(res1_fc1(x)));
        x = res1_bn2(res1_fc2(x));
        x = torch::relu(x + res);

        // Residual block 2
        res = x;
        x = torch::relu(res2_bn1(res2_fc1(x)));
        x = res2_bn2(res2_fc2(x));
        x = torch::relu(x + res);

        x = fc_out(x);
        return x;
    }
};
TORCH_MODULE(LiftingNet);
#endif  // HM_HAS_TORCH

struct DepthLifter::Impl {
    DepthLifterConfig config;
#ifdef HM_HAS_TORCH
    LiftingNet model;
#endif
    bool initialized = false;
    bool hasModel = false;

    std::array<Keypoint3D, COCO_KEYPOINTS> geometricFallback(
        const std::array<Keypoint2D, COCO_KEYPOINTS>& keypoints2D,
        const BBox& bbox) {

        std::array<Keypoint3D, COCO_KEYPOINTS> result{};

        // Estimate depth from torso length ratio
        // Using left/right shoulder (5,6) and hip (11,12) in COCO format

        float leftShoulderY = keypoints2D[5].position.y;
        float rightShoulderY = keypoints2D[6].position.y;
        float leftHipY = keypoints2D[11].position.y;
        float rightHipY = keypoints2D[12].position.y;

        float shoulderMidY = (leftShoulderY + rightShoulderY) * 0.5f;
        float hipMidY = (leftHipY + rightHipY) * 0.5f;

        float torsoLengthNorm = std::abs(shoulderMidY - hipMidY);
        if (torsoLengthNorm < 1e-4f) torsoLengthNorm = 0.3f;

        // Approximate real torso length: ~50cm for adults
        float scaleFactor = 50.0f / torsoLengthNorm;

        // Estimate depth as proportional to bbox height
        float bboxHeightNorm = bbox.height / std::max(1.0f, static_cast<float>(bbox.height));
        (void)bboxHeightNorm;
        float baseDepth = config.defaultSubjectHeight * 2.0f;

        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            result[k].position.x = (keypoints2D[k].position.x - 0.5f) * scaleFactor;
            result[k].position.y = -(keypoints2D[k].position.y - 0.5f) * scaleFactor;
            result[k].position.z = baseDepth + (keypoints2D[k].position.y - hipMidY) * scaleFactor * 0.1f;
            result[k].confidence = keypoints2D[k].confidence;
        }

        // Recenter around hip midpoint
        Vec3 hipMid{
            (result[11].position.x + result[12].position.x) * 0.5f,
            (result[11].position.y + result[12].position.y) * 0.5f,
            (result[11].position.z + result[12].position.z) * 0.5f
        };

        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            result[k].position.x -= hipMid.x;
            result[k].position.y -= hipMid.y;
            result[k].position.z -= hipMid.z;
        }

        return result;
    }
};

DepthLifter::DepthLifter(const DepthLifterConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

DepthLifter::~DepthLifter() = default;
DepthLifter::DepthLifter(DepthLifter&&) noexcept = default;
DepthLifter& DepthLifter::operator=(DepthLifter&&) noexcept = default;

bool DepthLifter::initialize() {
    try {
#ifdef HM_HAS_TORCH
        impl_->model = LiftingNet();

        if (!impl_->config.modelPath.empty()) {
            try {
                torch::serialize::InputArchive archive;
                archive.load_from(impl_->config.modelPath);
                impl_->model->load(archive);
                impl_->hasModel = true;
                HM_LOG_INFO(TAG, "Loaded lifting model from: " + impl_->config.modelPath);
            } catch (const std::exception& e) {
                HM_LOG_WARN(TAG, std::string("Could not load model: ") + e.what());
                if (!impl_->config.useGeometricFallback) {
                    return false;
                }
                HM_LOG_INFO(TAG, "Using geometric fallback for 2D->3D lifting");
            }
        } else {
            HM_LOG_INFO(TAG, "No model path specified, using geometric fallback");
        }

        impl_->model->eval();
#else
        HM_LOG_INFO(TAG, "LibTorch not available, using geometric fallback for 2D->3D lifting");
#endif
        impl_->initialized = true;
        return true;

    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Initialization failed: ") + e.what());
        return false;
    }
}

bool DepthLifter::isInitialized() const {
    return impl_->initialized;
}

bool DepthLifter::hasModel() const {
    return impl_->hasModel;
}

std::array<Keypoint3D, COCO_KEYPOINTS> DepthLifter::lift(
    const std::array<Keypoint2D, COCO_KEYPOINTS>& keypoints2D,
    const BBox& bbox) {

    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Not initialized");
        return {};
    }

    if (!impl_->hasModel) {
        return impl_->geometricFallback(keypoints2D, bbox);
    }

#ifdef HM_HAS_TORCH
    // Prepare input: 17 keypoints x 2D = 34, normalised relative to bbox centre
    std::vector<float> input(34);
    float cx = bbox.centerX();
    float cy = bbox.centerY();
    float scale = std::max(bbox.width, bbox.height);
    if (scale < 1e-4f) scale = 1.0f;

    for (int k = 0; k < COCO_KEYPOINTS; ++k) {
        input[k * 2 + 0] = (keypoints2D[k].position.x - cx) / scale;
        input[k * 2 + 1] = (keypoints2D[k].position.y - cy) / scale;
    }

    torch::NoGradGuard noGrad;
    auto tensor = torch::from_blob(input.data(), {1, 34}, torch::kFloat32).clone();
    auto output = impl_->model->forward(tensor);
    auto outputData = output.accessor<float, 2>();

    std::array<Keypoint3D, COCO_KEYPOINTS> result{};
    float heightScale = impl_->config.defaultSubjectHeight / 175.0f;

    for (int k = 0; k < COCO_KEYPOINTS; ++k) {
        result[k].position.x = outputData[0][k * 3 + 0] * heightScale;
        result[k].position.y = outputData[0][k * 3 + 1] * heightScale;
        result[k].position.z = outputData[0][k * 3 + 2] * heightScale;
        result[k].confidence = keypoints2D[k].confidence;
    }

    return result;
#else
    return impl_->geometricFallback(keypoints2D, bbox);
#endif
}

std::vector<std::array<Keypoint3D, COCO_KEYPOINTS>> DepthLifter::liftBatch(
    const std::vector<std::array<Keypoint2D, COCO_KEYPOINTS>>& keypoints2DList,
    const std::vector<BBox>& bboxes) {

    if (!impl_->initialized) return {};

    if (!impl_->hasModel) {
        std::vector<std::array<Keypoint3D, COCO_KEYPOINTS>> results;
        results.reserve(keypoints2DList.size());
        for (size_t i = 0; i < keypoints2DList.size(); ++i) {
            results.push_back(impl_->geometricFallback(keypoints2DList[i],
                              i < bboxes.size() ? bboxes[i] : BBox{}));
        }
        return results;
    }

#ifdef HM_HAS_TORCH
    int batchSize = static_cast<int>(keypoints2DList.size());
    std::vector<float> inputBatch(batchSize * 34);

    for (int b = 0; b < batchSize; ++b) {
        const auto& bbox = b < static_cast<int>(bboxes.size()) ? bboxes[b] : BBox{};
        float cx = bbox.centerX();
        float cy = bbox.centerY();
        float scale = std::max(bbox.width, bbox.height);
        if (scale < 1e-4f) scale = 1.0f;

        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            inputBatch[b * 34 + k * 2 + 0] = (keypoints2DList[b][k].position.x - cx) / scale;
            inputBatch[b * 34 + k * 2 + 1] = (keypoints2DList[b][k].position.y - cy) / scale;
        }
    }

    torch::NoGradGuard noGrad;
    auto tensor = torch::from_blob(inputBatch.data(), {batchSize, 34}, torch::kFloat32).clone();
    auto output = impl_->model->forward(tensor);
    auto outputData = output.accessor<float, 2>();

    float heightScale = impl_->config.defaultSubjectHeight / 175.0f;

    std::vector<std::array<Keypoint3D, COCO_KEYPOINTS>> results(batchSize);
    for (int b = 0; b < batchSize; ++b) {
        for (int k = 0; k < COCO_KEYPOINTS; ++k) {
            results[b][k].position.x = outputData[b][k * 3 + 0] * heightScale;
            results[b][k].position.y = outputData[b][k * 3 + 1] * heightScale;
            results[b][k].position.z = outputData[b][k * 3 + 2] * heightScale;
            results[b][k].confidence = keypoints2DList[b][k].confidence;
        }
    }

    return results;
#else
    std::vector<std::array<Keypoint3D, COCO_KEYPOINTS>> results;
    results.reserve(keypoints2DList.size());
    for (size_t i = 0; i < keypoints2DList.size(); ++i) {
        results.push_back(impl_->geometricFallback(keypoints2DList[i],
                          i < bboxes.size() ? bboxes[i] : BBox{}));
    }
    return results;
#endif
}

} // namespace hm::pose
