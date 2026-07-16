#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace offline_asr {

/**
 * CTC 解码器封装 — 封装 CTC 库的 PrefixBeamSearch
 */
class CtcDecoderWrapper {
public:
    struct Config {
        int beam_size = 10;
        int blank_id = 0;
        int n_best = 1;
        float cutoff_threshold = 0.0f;
    };

    struct Result {
        std::vector<int> tokens;
        float score;
        std::vector<int> timestamps;
    };

    explicit CtcDecoderWrapper(const Config& cfg);
    ~CtcDecoderWrapper();

    /** 非流式解码 */
    std::vector<Result> Decode(const float* log_probs, int num_frames, int vocab_size);

    /** 流式逐帧输入 */
    void Step(const float* log_probs, int vocab_size);

    /** 获取当前 n_best 结果 */
    std::vector<Result> Results(int n) const;

    void Reset();

private:
    Config cfg_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace offline_asr
