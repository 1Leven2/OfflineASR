#include "offline_asr/recognizer/recognizer_config.h"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace offline_asr {

RecognizerConfig RecognizerConfig::FromYaml(const std::string& path) {
    RecognizerConfig cfg;
    try {
        YAML::Node root = YAML::LoadFile(path);

        if (auto model = root["model"]) {
            cfg.model_path = model["path"].as<std::string>("");
            cfg.input_name = model["input_name"].as<std::string>("input");
            cfg.output_name = model["output_name"].as<std::string>("output");
        }

        if (auto runtime = root["runtime"]) {
            cfg.runtime_backend = runtime["backend"].as<std::string>("onnx");
            cfg.num_threads = runtime["num_threads"].as<int>(0);
        }

        if (auto feature = root["feature"]) {
            cfg.sample_rate = feature["sample_rate"].as<int>(16000);
            cfg.frame_length_ms = feature["frame_length_ms"].as<float>(25.0f);
            cfg.frame_shift_ms = feature["frame_shift_ms"].as<float>(10.0f);
            cfg.num_mel_bins = feature["num_mel_bins"].as<int>(80);
            cfg.pre_emphasis_alpha = feature["pre_emphasis_alpha"].as<float>(0.97f);
        }

        if (auto decoder = root["decoder"]) {
            cfg.beam_size = decoder["beam_size"].as<int>(10);
            cfg.n_best = decoder["n_best"].as<int>(1);
            cfg.cutoff_threshold = decoder["cutoff_threshold"].as<float>(0.0f);
        }

        if (auto vad = root["vad"]) {
            cfg.vad.enabled = vad["enabled"].as<bool>(false);
            cfg.vad.type = vad["type"].as<std::string>("energy");
            cfg.vad.threshold = vad["threshold"].as<float>(0.01f);
            cfg.vad.frame_ms = vad["frame_ms"].as<int>(30);
            cfg.vad.min_speech_ms = vad["min_speech_ms"].as<int>(100);
            cfg.vad.min_silence_ms = vad["min_silence_ms"].as<int>(200);
        }

        cfg.tokens_path = root["tokens"].as<std::string>("");
    } catch (const YAML::Exception& e) {
        spdlog::error("Failed to parse config YAML: {} ({})", path, e.what());
    }

    return cfg;
}

}  // namespace offline_asr
