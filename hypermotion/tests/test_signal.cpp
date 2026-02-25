#include <gtest/gtest.h>
#include "HyperMotion/signal/OutlierFilter.h"
#include "HyperMotion/signal/SavitzkyGolay.h"
#include "HyperMotion/signal/ButterworthFilter.h"
#include "HyperMotion/signal/QuaternionSmoother.h"
#include "HyperMotion/signal/FootContactFilter.h"
#include "HyperMotion/signal/SignalPipeline.h"
#include "HyperMotion/core/MathUtils.h"
#include "test_helpers.h"
#include <cmath>
#include <numeric>

using namespace hm;
using namespace hm::signal;

// ---------------------------------------------------------------
// OutlierFilter
// ---------------------------------------------------------------

TEST(OutlierFilterTest, FilterChannel_RemovesSpike) {
    // Create a clean signal with one spike
    std::vector<float> signal(50, 10.0f);
    signal[25] = 1000.0f; // outlier

    OutlierFilter::filterChannel(signal, 7, 3.0f);

    // The spike should be replaced with something near 10
    EXPECT_NEAR(signal[25], 10.0f, 1.0f);
    // Non-outlier values should remain unchanged
    EXPECT_FLOAT_EQ(signal[0], 10.0f);
    EXPECT_FLOAT_EQ(signal[49], 10.0f);
}

TEST(OutlierFilterTest, FilterChannel_PreservesCleanSignal) {
    std::vector<float> signal(30);
    for (int i = 0; i < 30; ++i)
        signal[i] = static_cast<float>(i);

    std::vector<float> original = signal;
    OutlierFilter::filterChannel(signal, 7, 3.0f);

    // A linearly increasing signal should not be modified
    for (size_t i = 0; i < signal.size(); ++i)
        EXPECT_NEAR(signal[i], original[i], 0.5f);
}

TEST(OutlierFilterTest, ProcessFrames) {
    auto frames = test::makeWalkingSequence(30);
    test::injectOutlier(frames, 15, 5, {500, 500, 500}); // Head spike

    OutlierFilter filter;
    filter.process(frames);

    // The spike should be attenuated
    // Head world position should be reasonable (not 500+ offset)
    float headY = frames[15].joints[5].worldPosition.y;
    EXPECT_LT(std::abs(headY), 300.0f); // Should be roughly near neighbors
}

// ---------------------------------------------------------------
// SavitzkyGolay
// ---------------------------------------------------------------

TEST(SavitzkyGolayTest, Coefficients_SumToOne) {
    auto coeffs = SavitzkyGolay::computeCoefficients(7, 3);
    float sum = 0.0f;
    for (float c : coeffs) sum += c;
    EXPECT_NEAR(sum, 1.0f, test::kEps);
}

TEST(SavitzkyGolayTest, Coefficients_Symmetric) {
    auto coeffs = SavitzkyGolay::computeCoefficients(7, 3);
    int n = static_cast<int>(coeffs.size());
    for (int i = 0; i < n / 2; ++i) {
        EXPECT_NEAR(coeffs[i], coeffs[n - 1 - i], test::kEps)
            << "Coefficient " << i << " vs " << (n - 1 - i);
    }
}

TEST(SavitzkyGolayTest, FilterChannel_SmoothsNoise) {
    // Noisy sinusoid
    std::vector<float> signal(100);
    std::vector<float> clean(100);
    for (int i = 0; i < 100; ++i) {
        float t = static_cast<float>(i) / 100.0f;
        clean[i] = std::sin(2.0f * 3.14159f * t);
        signal[i] = clean[i] + 0.1f * (((i * 7 + 13) % 37) / 37.0f - 0.5f);
    }

    SavitzkyGolay::filterChannel(signal, 7, 3);

    // Filtered signal should be closer to clean than noisy was
    float errorAfter = 0.0f;
    for (int i = 3; i < 97; ++i) // Avoid boundary effects
        errorAfter += (signal[i] - clean[i]) * (signal[i] - clean[i]);

    EXPECT_LT(errorAfter / 94.0f, 0.02f); // Low MSE
}

TEST(SavitzkyGolayTest, FilterChannel_PreservesConstant) {
    std::vector<float> signal(50, 42.0f);
    SavitzkyGolay::filterChannel(signal, 7, 3);
    for (float v : signal)
        EXPECT_NEAR(v, 42.0f, test::kEps);
}

// ---------------------------------------------------------------
// ButterworthFilter
// ---------------------------------------------------------------

TEST(ButterworthFilterTest, DesignLowpass_CoefficientsNonZero) {
    auto coeffs = ButterworthFilter::designLowpass(4, 12.0f, 30.0f);
    EXPECT_FALSE(coeffs.b.empty());
    EXPECT_FALSE(coeffs.a.empty());
    // a[0] should be 1 (normalized)
    EXPECT_NEAR(coeffs.a[0], 1.0f, test::kEps);
}

TEST(ButterworthFilterTest, FilterChannel_SmoothsStep) {
    // Step function
    std::vector<float> signal(100, 0.0f);
    for (int i = 50; i < 100; ++i)
        signal[i] = 1.0f;

    ButterworthFilter::filterChannel(signal, 4, 5.0f, 30.0f);

    // After filtering, the step should be smooth (no sharp transition)
    // Check that mid-transition values are between 0 and 1
    EXPECT_GT(signal[50], -0.1f);
    EXPECT_LT(signal[50], 1.1f);
    // Far from transition should be near original values
    EXPECT_NEAR(signal[10], 0.0f, 0.1f);
    EXPECT_NEAR(signal[90], 1.0f, 0.1f);
}

TEST(ButterworthFilterTest, FilterChannel_PreservesConstant) {
    std::vector<float> signal(60, 7.5f);
    ButterworthFilter::filterChannel(signal, 4, 12.0f, 30.0f);
    for (float v : signal)
        EXPECT_NEAR(v, 7.5f, test::kEps);
}

// ---------------------------------------------------------------
// QuaternionSmoother
// ---------------------------------------------------------------

TEST(QuaternionSmootherTest, SmootherNoChange) {
    // All identical quaternions should remain unchanged
    std::vector<Quat> quats(20, Quat::identity());
    QuaternionSmoother::smoothQuatSequence(quats, 0.3f);
    for (const auto& q : quats)
        EXPECT_TRUE(test::quatNearEqual(q, Quat::identity()));
}

TEST(QuaternionSmootherTest, SmootherReducesNoise) {
    // Create a smooth rotation sequence with a spike
    std::vector<Quat> quats(30);
    for (int i = 0; i < 30; ++i)
        quats[i] = MathUtils::fromAxisAngle({0, 1, 0}, static_cast<float>(i) * 2.0f);

    // Inject a large spike
    quats[15] = MathUtils::fromAxisAngle({0, 1, 0}, 180.0f);

    QuaternionSmoother::smoothQuatSequence(quats, 0.3f);

    // After smoothing, the spike should be reduced
    // The neighbors should still be close to their original values
    Quat expected14 = MathUtils::fromAxisAngle({0, 1, 0}, 28.0f);
    EXPECT_TRUE(test::quatNearEqual(quats[14], expected14, 0.3f));
}

// ---------------------------------------------------------------
// FootContactFilter
// ---------------------------------------------------------------

TEST(FootContactFilterTest, DetectsStationaryFoot) {
    auto frames = test::makeWalkingSequence(30);
    // Make left foot stationary on ground
    for (int i = 0; i < 30; ++i) {
        int leftFoot = static_cast<int>(Joint::LeftFoot);
        frames[i].joints[leftFoot].worldPosition = {0, 0, 0}; // On ground
    }

    FootContactFilter filter;
    auto contacts = filter.detectContacts(frames);

    EXPECT_EQ(contacts.size(), frames.size());
    // Most frames should detect left foot contact (velocity ~0, height ~0)
    int leftContactCount = 0;
    for (const auto& c : contacts)
        if (c.leftFootContact) leftContactCount++;
    EXPECT_GT(leftContactCount, 20); // Most frames should detect contact
}

TEST(FootContactFilterTest, NoContactWhenMoving) {
    auto frames = test::makeWalkingSequence(30);
    // Make feet move rapidly and high off ground
    for (int i = 0; i < 30; ++i) {
        int leftFoot = static_cast<int>(Joint::LeftFoot);
        frames[i].joints[leftFoot].worldPosition = {
            static_cast<float>(i) * 10.0f, 50.0f, 0.0f};
    }

    FootContactFilter filter;
    auto contacts = filter.detectContacts(frames);

    int leftContactCount = 0;
    for (const auto& c : contacts)
        if (c.leftFootContact) leftContactCount++;
    EXPECT_LT(leftContactCount, 5); // Few or no contacts
}

// ---------------------------------------------------------------
// SignalPipeline
// ---------------------------------------------------------------

TEST(SignalPipelineTest, ProcessDoesNotCrash) {
    auto frames = test::makeWalkingSequence(60);
    SignalPipeline pipeline;
    EXPECT_NO_THROW(pipeline.process(frames));
    EXPECT_EQ(frames.size(), 60u);
}

TEST(SignalPipelineTest, StageEnableDisable) {
    auto frames = test::makeWalkingSequence(30);
    auto original = frames;

    SignalPipelineConfig config;
    config.enableOutlierFilter = false;
    config.enableSavitzkyGolay = false;
    config.enableButterworth = false;
    config.enableQuaternionSmoothing = false;
    config.enableFootContact = false;

    SignalPipeline pipeline(config);
    pipeline.process(frames);

    // With all stages disabled, frames should be unchanged
    for (size_t i = 0; i < frames.size(); ++i) {
        EXPECT_NEAR(frames[i].rootPosition.x, original[i].rootPosition.x, test::kEps);
    }
}
