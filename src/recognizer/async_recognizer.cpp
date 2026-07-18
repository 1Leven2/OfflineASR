#include "offline_asr/recognizer/async_recognizer.h"

#include "offline_asr/audio/wav_reader.h"
#include "offline_asr/decoder/ctc_wrapper.h"
#include "offline_asr/decoder/token_table.h"
#include "offline_asr/feature/fbank_extractor.h"
#include "offline_asr/runtime/runtime_factory.h"
#include "offline_asr/utils/thread_pool.h"

#include <spdlog/spdlog.h>
#include <atomic>

namespace offline_asr {

class AsyncRecognizer::Impl {
public:
    Impl(const RecognizerConfig& cfg, int num_workers)
        : cfg_(cfg), pool_(num_workers), pending_(0) {}

    bool Init() {
        runtime_ = RuntimeFactory::Create(cfg_.runtime_backend, cfg_);
        if (!runtime_) return false;
        if (!runtime_->LoadModel(cfg_.model_path, cfg_.input_name, cfg_.output_name))
            return false;

        if (!cfg_.tokens_path.empty()) {
            if (!token_table_.Load(cfg_.tokens_path)) {
                token_table_ = CreateCharVocab();
            }
        } else {
            token_table_ = CreateCharVocab();
        }

        spdlog::info("AsyncRecognizer: {} workers, backend={}",
                     pool_.NumThreads(), runtime_->BackendName());
        return true;
    }

    std::future<RecognitionResult> RecognizeAsync(const float* samples, size_t n) {
        // Copy samples — the caller's buffer may not outlive the task
        std::vector<float> audio(samples, samples + n);
        pending_.fetch_add(1, std::memory_order_release);

        return pool_.Enqueue([this, audio = std::move(audio)]() -> RecognitionResult {
            RecognitionResult result;
            size_t num_samples = audio.size();

            // Preprocess: DC removal + peak normalize
            std::vector<float> buffer(num_samples);
            float mean = 0.0F;
            for (size_t i = 0; i < num_samples; ++i)
                mean += audio[i];
            mean /= static_cast<float>(num_samples);

            float peak = 0.0F;
            for (size_t i = 0; i < num_samples; ++i) {
                float v = audio[i] - mean;
                buffer[i] = v;
                float abs_v = v < 0.0F ? -v : v;
                if (abs_v > peak) peak = abs_v;
            }
            if (peak > 0.0F) {
                float inv_peak = 1.0F / peak;
                for (size_t i = 0; i < num_samples; ++i)
                    buffer[i] *= inv_peak;
            }

            double audio_duration = static_cast<double>(num_samples) /
                                    cfg_.sample_rate * 1000.0;
            result.audio_duration_ms = audio_duration;

            // FBank (local extractor per task)
            FbankExtractor fbank_extractor({cfg_.sample_rate, cfg_.frame_length_ms,
                                            cfg_.frame_shift_ms, cfg_.num_mel_bins,
                                            cfg_.pre_emphasis_alpha, true});
            auto fbank = fbank_extractor.Extract(buffer.data(), num_samples);
            int num_frames = fbank_extractor.NumFrames(num_samples);
            if (fbank.empty() || num_frames == 0) {
                pending_.fetch_sub(1, std::memory_order_release);
                return result;
            }

            // Inference (shared runtime, thread-safe)
            auto log_probs = runtime_->Forward(fbank.data(), num_frames, cfg_.num_mel_bins);
            if (log_probs.empty()) {
                pending_.fetch_sub(1, std::memory_order_release);
                return result;
            }

            int vocab_size = static_cast<int>(runtime_->VocabSize());
            int output_frames = static_cast<int>(log_probs.size()) / vocab_size;
            if (output_frames == 0) {
                pending_.fetch_sub(1, std::memory_order_release);
                return result;
            }

            // Decode (local decoder per task)
            CtcDecoderWrapper::Config decoder_cfg;
            decoder_cfg.beam_size = cfg_.beam_size;
            decoder_cfg.n_best = cfg_.n_best;
            decoder_cfg.cutoff_threshold = cfg_.cutoff_threshold;
            CtcDecoderWrapper decoder(decoder_cfg);

            auto decode_results = decoder.Decode(log_probs.data(), output_frames, vocab_size);
            if (!decode_results.empty()) {
                result.tokens = decode_results[0].tokens;
                result.confidence = decode_results[0].score;
                result.text = token_table_.Decode(result.tokens);
            }

            pending_.fetch_sub(1, std::memory_order_release);
            return result;
        });
    }

    std::future<RecognitionResult> RecognizeAsync(const std::string& audio_path) {
        auto audio = WavReader::Read(audio_path);
        if (audio.samples.empty()) {
            std::promise<RecognitionResult> p;
            p.set_value({});
            return p.get_future();
        }
        return RecognizeAsync(audio.samples.data(), audio.samples.size());
    }

    void WaitAll() {
        auto f = pool_.Enqueue([] {});
        f.wait();
    }

    size_t Pending() const {
        return pending_.load(std::memory_order_acquire);
    }

private:
    RecognizerConfig cfg_;
    ThreadPool pool_;
    std::unique_ptr<Runtime> runtime_;
    TokenTable token_table_;
    std::atomic<size_t> pending_;
};

AsyncRecognizer::AsyncRecognizer(const RecognizerConfig& cfg, int num_workers)
    : impl_(std::make_unique<Impl>(cfg, num_workers)) {
    impl_->Init();
}

AsyncRecognizer::AsyncRecognizer(const std::string& yaml_path, int num_workers)
    : impl_(std::make_unique<Impl>(RecognizerConfig::FromYaml(yaml_path), num_workers)) {
    impl_->Init();
}

AsyncRecognizer::~AsyncRecognizer() = default;

std::future<RecognitionResult> AsyncRecognizer::RecognizeAsync(const std::string& audio_path) {
    return impl_->RecognizeAsync(audio_path);
}

std::future<RecognitionResult> AsyncRecognizer::RecognizeAsync(const float* samples, size_t n) {
    return impl_->RecognizeAsync(samples, n);
}

void AsyncRecognizer::WaitAll() { impl_->WaitAll(); }
size_t AsyncRecognizer::Pending() const { return impl_->Pending(); }

}  // namespace offline_asr
