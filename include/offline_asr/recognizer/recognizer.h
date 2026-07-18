#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "offline_asr/recognizer/recognizer_config.h"

namespace offline_asr {

struct RecognitionResult {
    std::string text;
    float confidence = 0.0f;
    std::vector<int> tokens;
    std::vector<float> timestamps;
    double audio_duration_ms = 0.0;
    double rtf = 0.0;  // real-time factor

    // Per-module latency breakdown (ms)
    double preprocess_ms = 0.0;
    double fbank_ms = 0.0;
    double inference_ms = 0.0;
    double decoder_ms = 0.0;
};

/**
 * 语音识别器 — 系统唯一对外入口
 *
 * 使用示例:
 * @code
 *   Recognizer recognizer("config.yaml");
 *   auto result = recognizer.Recognize("demo.wav");
 *   std::cout << result.text << std::endl;
 * @endcode
 *
 * 内部流程:
 *   Audio → FBank → Runtime → CTC Decoder → Text
 */
class Recognizer {
public:
    explicit Recognizer(const RecognizerConfig& cfg);

    /** 从 YAML 配置文件构造 */
    explicit Recognizer(const std::string& yaml_path);

    ~Recognizer();

    /** 离线识别音频文件 */
    RecognitionResult Recognize(const std::string& audio_path);

    /** 离线识别内存中的音频采样 */
    RecognitionResult Recognize(const float* samples, size_t num_samples);

    // 流式接口 (V1.1 预留)
    void AcceptWaveform(const float* samples, size_t num_samples);
    std::string Decode();
    std::string GetPartialResult();
    void Reset();

    /** Returns true if endpoint detection triggered (streaming mode only) */
    bool IsEndpoint();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace offline_asr
