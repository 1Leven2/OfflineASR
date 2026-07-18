#include <spdlog/spdlog.h>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "offline_asr/audio/wav_reader.h"
#include "offline_asr/recognizer/async_recognizer.h"
#include "offline_asr/recognizer/recognizer.h"
#include "offline_asr/utils/logger.h"

using namespace offline_asr;

static void print_usage() {
    std::cerr << "Usage: offline_asr <command> [options]\n\n"
              << "Commands:\n"
              << "  transcribe <audio>     [-c config.yaml]\n"
              << "  batch <filelist>       [-c config.yaml] [-o output.txt]\n"
              << "  batch_async <filelist> [-c config.yaml] [-o output.txt] [-j N]\n"
              << "  stream <audio>         [-c config.yaml] [-k chunk_ms]\n"
              << "  benchmark              [-c config.yaml] [-i iterations]\n"
              << "  benchmark_parallel     [-c config.yaml] [-i iterations] [-j N]\n";
}

static const char* get_arg(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return nullptr;
}

int cmd_transcribe(const std::string& audio_path, const std::string& config_path) {
    InitLogger("info");

    Recognizer recognizer(config_path);
    auto result = recognizer.Recognize(audio_path);

    if (result.text.empty()) {
        spdlog::error("Recognition failed or produced empty result");
        return 1;
    }

    std::cout << "Text:       " << result.text << "\n";
    std::cout << "Confidence: " << result.confidence << "\n";
    std::cout << "Duration:   " << result.audio_duration_ms << " ms\n";
    std::cout << "RTF:        " << result.rtf << "\n";

    return 0;
}

int cmd_batch(const std::string& filelist, const std::string& config_path,
              const std::string& output_path) {
    InitLogger("info");

    std::ifstream list(filelist);
    if (!list.is_open()) {
        spdlog::error("Failed to open file list: {}", filelist);
        return 1;
    }

    std::ofstream out;
    if (!output_path.empty()) {
        out.open(output_path);
    }

    Recognizer recognizer(config_path);
    std::string line;
    int count = 0;
    while (std::getline(list, line)) {
        if (line.empty() || line[0] == '#') continue;
        ++count;

        auto result = recognizer.Recognize(line);
        std::string text = result.text.empty() ? "<EMPTY>" : result.text;

        if (out.is_open()) {
            out << line << "\t" << text << "\n";
        } else {
            std::cout << "[" << count << "] " << line << " -> " << text << "\n";
        }
    }

    if (out.is_open()) {
        spdlog::info("Results written to {}", output_path);
    }
    return 0;
}

int cmd_benchmark(const std::string& config_path, int iterations) {
    InitLogger("warn");

    Recognizer recognizer(config_path);

    std::vector<float> audio(16000);
    for (int i = 0; i < 16000; ++i) {
        audio[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }

    // Warm-up
    recognizer.Recognize(audio.data(), audio.size());

    double total_ms = 0.0;
    double preprocess_ms = 0.0, fbank_ms = 0.0, inference_ms = 0.0, decoder_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto result = recognizer.Recognize(audio.data(), audio.size());
        double t = result.preprocess_ms + result.fbank_ms +
                   result.inference_ms + result.decoder_ms;
        total_ms += t;
        preprocess_ms += result.preprocess_ms;
        fbank_ms += result.fbank_ms;
        inference_ms += result.inference_ms;
        decoder_ms += result.decoder_ms;
    }

    double n = iterations;
    std::cout << "======================\n";
    std::cout << "Benchmark Results\n";
    std::cout << "======================\n";
    std::cout << "Iterations:      " << iterations << "\n";
    std::cout << "Avg Latency:     " << total_ms / n << " ms\n";
    std::cout << "RTF:             " << (total_ms / n) / 1000.0 << "\n";
    std::cout << "----------------------\n";
    std::cout << "Preprocess:      " << preprocess_ms / n << " ms\n";
    std::cout << "FBank:           " << fbank_ms / n << " ms\n";
    std::cout << "Inference:       " << inference_ms / n << " ms\n";
    std::cout << "Decoder:         " << decoder_ms / n << " ms\n";
    std::cout << "======================\n";

    return 0;
}

int cmd_batch_async(const std::string& filelist, const std::string& config_path,
                    const std::string& output_path, int num_workers) {
    InitLogger("info");

    std::ifstream list(filelist);
    if (!list.is_open()) {
        spdlog::error("Failed to open file list: {}", filelist);
        return 1;
    }

    // Read all lines first (need them for parallel dispatch)
    std::vector<std::string> files;
    std::string line;
    while (std::getline(list, line)) {
        if (line.empty() || line[0] == '#') continue;
        files.push_back(line);
    }

    if (files.empty()) {
        spdlog::warn("No files in list");
        return 0;
    }

    AsyncRecognizer recognizer(config_path, num_workers);

    std::ofstream out;
    if (!output_path.empty()) out.open(output_path);

    std::vector<std::pair<std::string, std::future<RecognitionResult>>> futures;
    futures.reserve(files.size());

    auto t0 = std::chrono::steady_clock::now();

    for (const auto& f : files) {
        futures.emplace_back(f, recognizer.RecognizeAsync(f));
    }

    for (auto& [path, fut] : futures) {
        auto result = fut.get();
        std::string text = result.text.empty() ? "<EMPTY>" : result.text;

        if (out.is_open()) {
            out << path << "\t" << text << "\n";
        } else {
            std::cout << path << " -> " << text << "\n";
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    spdlog::info("Processed {} files in {:.0f} ms ({:.1f} files/s)",
                 files.size(), elapsed, files.size() / (elapsed / 1000.0));

    if (out.is_open()) {
        spdlog::info("Results written to {}", output_path);
    }
    return 0;
}

int cmd_benchmark_parallel(const std::string& config_path, int iterations,
                           int num_workers) {
    InitLogger("warn");

    AsyncRecognizer recognizer(config_path, num_workers);

    // Synthetic 1s audio
    std::vector<float> audio(16000);
    for (int i = 0; i < 16000; ++i) {
        audio[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }

    // Warm-up
    {
        auto f = recognizer.RecognizeAsync(audio.data(), audio.size());
        f.get();
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::future<RecognitionResult>> futures;
    futures.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        futures.push_back(recognizer.RecognizeAsync(audio.data(), audio.size()));
    }
    for (auto& f : futures) {
        f.get();
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iterations;

    std::cout << "========================\n";
    std::cout << "Parallel Benchmark\n";
    std::cout << "========================\n";
    std::cout << "Workers:         " << num_workers << "\n";
    std::cout << "Iterations:      " << iterations << "\n";
    std::cout << "Total time:      " << total_ms << " ms\n";
    std::cout << "Avg latency:     " << avg_ms << " ms\n";
    std::cout << "Throughput:      " << (iterations / (total_ms / 1000.0))
              << " req/s\n";
    std::cout << "========================\n";

    return 0;
}

int cmd_stream(const std::string& audio_path, int chunk_ms,
               const std::string& config_path) {
    InitLogger("info");

    auto audio = WavReader::Read(audio_path);
    if (audio.samples.empty()) {
        spdlog::error("Failed to read audio: {}", audio_path);
        return 1;
    }

    Recognizer recognizer(config_path);
    size_t chunk_size = static_cast<size_t>(audio.sample_rate * chunk_ms / 1000);
    size_t offset = 0;
    int chunk_idx = 0;

    while (offset < audio.samples.size()) {
        size_t n = std::min(chunk_size, audio.samples.size() - offset);
        recognizer.AcceptWaveform(audio.samples.data() + offset, n);
        offset += n;

        auto partial = recognizer.GetPartialResult();
        if (!partial.empty()) {
            std::cout << "[" << ++chunk_idx << "] " << partial << std::endl;
        }
    }

    std::string final_text = recognizer.Decode();
    std::cout << "[final] " << final_text << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];
    const char* config = get_arg(argc, argv, "-c");
    std::string config_path = config ? config : "configs/default.yaml";

    if (command == "transcribe") {
        if (argc < 3) {
            std::cerr << "Usage: offline_asr transcribe <audio_file> [-c config.yaml]\n";
            return 1;
        }
        return cmd_transcribe(argv[2], config_path);
    }
    else if (command == "batch") {
        if (argc < 3) {
            std::cerr << "Usage: offline_asr batch <filelist> [-c config.yaml] [-o output.txt]\n";
            return 1;
        }
        const char* output = get_arg(argc, argv, "-o");
        std::string output_path = output ? output : "";
        return cmd_batch(argv[2], config_path, output_path);
    }
    else if (command == "batch_async") {
        if (argc < 3) {
            std::cerr << "Usage: offline_asr batch_async <filelist> [-c config.yaml] [-o output.txt] [-j N]\n";
            return 1;
        }
        const char* output = get_arg(argc, argv, "-o");
        std::string output_path = output ? output : "";
        const char* jobs = get_arg(argc, argv, "-j");
        int num_workers = jobs ? std::stoi(jobs) : 0;
        return cmd_batch_async(argv[2], config_path, output_path, num_workers);
    }
    else if (command == "stream") {
        if (argc < 3) {
            std::cerr << "Usage: offline_asr stream <audio_file> [-c config.yaml] [-k chunk_ms]\n";
            return 1;
        }
        const char* chunk_arg = get_arg(argc, argv, "-k");
        int chunk_ms = chunk_arg ? std::stoi(chunk_arg) : 400;
        return cmd_stream(argv[2], chunk_ms, config_path);
    }
    else if (command == "benchmark") {
        const char* iter_str = get_arg(argc, argv, "-i");
        int iterations = iter_str ? std::stoi(iter_str) : 100;
        return cmd_benchmark(config_path, iterations);
    }
    else if (command == "benchmark_parallel") {
        const char* iter_str = get_arg(argc, argv, "-i");
        int iterations = iter_str ? std::stoi(iter_str) : 100;
        const char* jobs = get_arg(argc, argv, "-j");
        int num_workers = jobs ? std::stoi(jobs) : 0;
        return cmd_benchmark_parallel(config_path, iterations, num_workers);
    }
    else {
        std::cerr << "Unknown command: " << command << "\n";
        print_usage();
        return 1;
    }
}
