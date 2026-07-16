#pragma once

#include <chrono>
#include <string>

namespace offline_asr {

/**
 * 高精度计时器 — 用于 Benchmark 和 Profiler
 *
 * 使用:
 *   Timer t;
 *   // ... 被测代码 ...
 *   double ms = t.ElapsedMs();
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    /** 重置计时器 */
    void Reset() { start_ = std::chrono::high_resolution_clock::now(); }

    /** 已过毫秒数 */
    double ElapsedMs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    /** 已过微秒数 */
    double ElapsedUs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(now - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace offline_asr
