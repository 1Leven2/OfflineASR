#pragma once

#include <cstddef>
#include <vector>

namespace offline_asr {

/**
 * 流式 FBank 特征提取器 — 逐块输入音频，逐帧输出特征
 *
 * 内部维护分帧缓冲区和预加重状态，支持任意大小的音频块输入。
 * 与 FbankExtractor (批处理) 对应，流式场景使用此类。
 *
 * 使用示例:
 * @code
 *   StreamingFbankExtractor extractor(cfg);
 *   while (has_audio()) {
 *     auto frames = extractor.Feed(chunk, chunk_size);
 *     // frames 布局: [new_frames x num_mel_bins] row-major
 *   }
 *   auto final_frames = extractor.Flush();
 * @endcode
 *
 * 线程安全: 非线程安全，调用方负责单线程访问。
 */
class StreamingFbankExtractor {
public:
    struct Config {
        int sample_rate = 16000;
        float frame_length_ms = 25.0f;
        float frame_shift_ms = 10.0f;
        int num_mel_bins = 80;
        float pre_emphasis_alpha = 0.97f;
        float log_floor = 1.1920928955078125e-07f;  // Kaldi epsilon
        float low_freq = 20.0f;
        float high_freq = 0.0f;  // 0 = auto Nyquist
    };

    explicit StreamingFbankExtractor(const Config& cfg);

    /**
     * 输入一批音频采样，返回新产生的 FBank 帧
     * @param samples  归一化音频采样 (DC 已去除, peak 已归一化)
     * @param n        采样数
     * @return 新帧 [N_new x num_mel_bins] row-major, 可能为空
     */
    std::vector<float> Feed(const float* samples, size_t n);

    /**
     * 冲刷剩余采样 (末尾补零凑够一帧)
     * @return 最后一帧，可能为空（如果没有剩余采样）
     */
    std::vector<float> Flush();

    /** 重置所有内部状态 (开始新语音段) */
    void Reset();

    int NumMelBins() const { return num_mel_bins_; }

private:
    std::vector<float> ProcessFrame(const float* frame) const;

    // Derived from Config in constructor
    int frame_length_;    // samples per frame (e.g. 400 for 25ms@16kHz)
    int frame_shift_;     // samples between frames (e.g. 160 for 10ms@16kHz)
    int n_fft_;           // FFT size (next pow2 of frame_length_)
    int num_fft_bins_;    // n_fft_ / 2 + 1
    int num_mel_bins_;
    float alpha_;         // pre-emphasis coefficient
    float log_floor_;

    // Pre-computed once in constructor
    std::vector<float> window_;      // length = frame_length_
    std::vector<float> mel_matrix_;   // [num_mel_bins_ x num_fft_bins_]

    // Streaming state
    float prev_sample_ = 0.0f;       // last raw sample for pre-emphasis continuity
    std::vector<float> buffer_;       // pre-emphasized samples awaiting framing
    size_t total_fed_ = 0;           // total samples fed
    int frames_emitted_ = 0;        // number of complete frames emitted so far
};

}  // namespace offline_asr
