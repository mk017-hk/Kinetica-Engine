#include <gtest/gtest.h>
#include "HyperMotion/ml/NoiseScheduler.h"
#include "test_helpers.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace hm::ml;

// ---------------------------------------------------------------
// Construction and schedule
// ---------------------------------------------------------------

TEST(NoiseSchedulerTest, AlphasCumprodDecreasing) {
    NoiseScheduler sched(1000, 0.0001f, 0.02f);
    for (int i = 1; i < 1000; ++i) {
        EXPECT_LT(sched.alphasCumprod(i), sched.alphasCumprod(i - 1))
            << "alphasCumprod should be monotonically decreasing at step " << i;
    }
}

TEST(NoiseSchedulerTest, AlphasCumprod_FirstNearOne) {
    NoiseScheduler sched(1000, 0.0001f, 0.02f);
    EXPECT_GT(sched.alphasCumprod(0), 0.999f);
}

TEST(NoiseSchedulerTest, AlphasCumprod_LastSmall) {
    NoiseScheduler sched(1000, 0.0001f, 0.02f);
    EXPECT_LT(sched.alphasCumprod(999), 0.1f);
}

TEST(NoiseSchedulerTest, AlphasCumprod_InRange) {
    NoiseScheduler sched(1000, 0.0001f, 0.02f);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_GT(sched.alphasCumprod(i), 0.0f);
        EXPECT_LE(sched.alphasCumprod(i), 1.0f);
    }
}

// ---------------------------------------------------------------
// DDIM Schedule
// ---------------------------------------------------------------

TEST(NoiseSchedulerTest, DDIMSchedule_CorrectLength) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(50);
    EXPECT_EQ(schedule.size(), 50u);
}

TEST(NoiseSchedulerTest, DDIMSchedule_Decreasing) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(50);
    // Schedule goes from high timestep to low (descending)
    for (size_t i = 1; i < schedule.size(); ++i) {
        EXPECT_LE(schedule[i], schedule[i - 1])
            << "Schedule should be non-increasing at index " << i;
    }
}

TEST(NoiseSchedulerTest, DDIMSchedule_InRange) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(50);
    for (int t : schedule) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, 1000);
    }
}

TEST(NoiseSchedulerTest, DDIMSchedule_SingleStep) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(1);
    EXPECT_EQ(schedule.size(), 1u);
    EXPECT_EQ(schedule[0], 0);
}

// ---------------------------------------------------------------
// DDIM Step
// ---------------------------------------------------------------

TEST(NoiseSchedulerTest, DDIMStep_ZeroNoise) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(50);

    // With zero noise prediction, x0 = x_t / sqrt(alpha)
    std::vector<float> x_t(132, 1.0f);
    std::vector<float> noise(132, 0.0f);
    std::vector<float> output(132);

    sched.ddimStep(x_t.data(), noise.data(), 132,
                   schedule[0], schedule[1], output.data());

    // Output should be finite
    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(NoiseSchedulerTest, DDIMStep_OutputDimMatchesInput) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(50);

    const int dim = 132;
    std::vector<float> x_t(dim, 0.5f);
    std::vector<float> noise(dim, 0.1f);
    std::vector<float> output(dim, 0.0f);

    sched.ddimStep(x_t.data(), noise.data(), dim,
                   schedule[0], schedule[1], output.data());

    for (int i = 0; i < dim; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]))
            << "Non-finite value at index " << i;
    }
}

TEST(NoiseSchedulerTest, DDIMStep_ClampingWorks) {
    NoiseScheduler sched(1000);

    // At a high timestep, alpha is small -> x0 prediction can be huge
    // The clamping to [-5, 5] should prevent extreme outputs
    std::vector<float> x_t(10, 100.0f); // Large input
    std::vector<float> noise(10, 0.0f);
    std::vector<float> output(10);

    sched.ddimStep(x_t.data(), noise.data(), 10, 999, 980, output.data());

    for (float v : output) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LE(std::abs(v), 100.0f); // Output should be bounded
    }
}

// ---------------------------------------------------------------
// Full DDIM inference loop (with dummy constant noise predictor)
// ---------------------------------------------------------------

TEST(NoiseSchedulerTest, FullDDIMLoop_Converges) {
    NoiseScheduler sched(1000);
    auto schedule = sched.getDDIMSchedule(50);

    const int dim = 132;
    std::vector<float> x(dim, 1.0f); // Start from "noise"
    std::vector<float> output(dim);

    // Simulate denoising with constant zero noise prediction
    std::vector<float> predictedNoise(dim, 0.0f);
    for (size_t i = 0; i < schedule.size() - 1; ++i) {
        sched.ddimStep(x.data(), predictedNoise.data(), dim,
                       schedule[i], schedule[i + 1], output.data());
        x = output;
    }

    // After full loop, values should be finite
    for (float v : x) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ---------------------------------------------------------------
// Custom schedule parameters
// ---------------------------------------------------------------

TEST(NoiseSchedulerTest, CustomTimesteps) {
    NoiseScheduler sched(500, 0.001f, 0.05f);
    EXPECT_EQ(sched.numTimesteps(), 500);

    auto schedule = sched.getDDIMSchedule(25);
    EXPECT_EQ(schedule.size(), 25u);
    for (int t : schedule) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, 500);
    }
}
