#pragma once

#include <string>
#include <vector>

namespace offline_asr {

struct VadConfig {
    bool enabled = false;
    std::string type = "energy";
    float threshold = 0.01f;
    int frame_ms = 30;
    int min_speech_ms = 100;
    int min_silence_ms = 200;
};

struct TensorRtConfig {
    int device_id = 0;
    bool use_fp16 = false;
    size_t max_workspace_size = 1ULL << 30;
    std::string engine_cache_dir = "models";
};

/**
 * 识别器配置 — 从 YAML 文件解析或代码构建
 */
struct RecognizerConfig {
    // Audio
    int sample_rate = 16000;

    // Feature
    int num_mel_bins = 80;
    float frame_length_ms = 25.0f;
    float frame_shift_ms = 10.0f;
    float pre_emphasis_alpha = 0.97f;

    // Runtime
    std::string model_path;
    std::string runtime_backend = "onnx";
    std::string input_name = "input";
    std::string output_name = "output";
    int num_threads = 0;

    // TensorRT
    TensorRtConfig tensorrt;

    // Decoder
    int beam_size = 10;
    int n_best = 1;
    float cutoff_threshold = 0.0f;

    // Tokens
    std::string tokens_path;

    // VAD
    VadConfig vad;

    /** 从 YAML 文件加载配置 */
    static RecognizerConfig FromYaml(const std::string& path);
};

}  // namespace offline_asr
