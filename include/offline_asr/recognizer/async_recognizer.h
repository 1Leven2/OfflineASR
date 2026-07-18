#pragma once

#include <cstddef>
#include <future>
#include <memory>
#include <string>

#include "offline_asr/recognizer/recognizer_config.h"
#include "offline_asr/recognizer/recognizer.h"

namespace offline_asr {

class ThreadPool;

class AsyncRecognizer {
public:
    explicit AsyncRecognizer(const RecognizerConfig& cfg, int num_workers = 0);
    explicit AsyncRecognizer(const std::string& yaml_path, int num_workers = 0);
    ~AsyncRecognizer();

    AsyncRecognizer(const AsyncRecognizer&) = delete;
    AsyncRecognizer& operator=(const AsyncRecognizer&) = delete;

    std::future<RecognitionResult> RecognizeAsync(const std::string& audio_path);
    std::future<RecognitionResult> RecognizeAsync(const float* samples, size_t n);

    void WaitAll();
    size_t Pending() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace offline_asr
