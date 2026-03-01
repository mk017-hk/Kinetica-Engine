#pragma once

#include <chrono>
#include <string>

namespace hm {

/// Lightweight RAII timer. Records wall-clock elapsed time in milliseconds.
/// Usage:
///   double elapsed = 0;
///   { ScopedTimer t(elapsed); doWork(); }
///   std::cout << "took " << elapsed << " ms\n";
class ScopedTimer {
public:
    explicit ScopedTimer(double& outMs)
        : outMs_(outMs), start_(Clock::now()) {}

    ~ScopedTimer() {
        auto end = Clock::now();
        outMs_ = std::chrono::duration<double, std::milli>(end - start_).count();
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /// Query elapsed time so far (before destruction).
    double elapsedMs() const {
        auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    double& outMs_;
    Clock::time_point start_;
};

} // namespace hm
