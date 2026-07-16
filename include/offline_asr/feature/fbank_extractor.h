#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace offline_asr {

/**
 * FBank 特征提取器 — 封装 FBank 库的 compute_fbank
 *
 * 流水线: 音频采样 → 预加重 → 分帧 → 加窗 → FFT → Mel滤波器组 → Log → 特征
 */
class FbankExtractor {
public:
    struct Config {
        int32_t sample_rate = 16000;
        float frame_length_ms = 25.0f;
        float frame_shift_ms = 10.0f;
        int32_t num_mel_bins = 80;
        float pre_emphasis_alpha = 0.97f;
        bool use_log = true;
    };

    explicit FbankExtractor(const Config& cfg);

    /** 从归一化音频采样提取特征，返回 [num_frames, num_mel_bins] row-major */
    std::vector<float> Extract(const float* samples, size_t num_samples);

    /** 预计算帧数 */
    int32_t NumFrames(size_t num_samples) const;

    int32_t NumMelBins() const { return cfg_.num_mel_bins; }

private:
    Config cfg_;
};

}  // namespace offline_asr
