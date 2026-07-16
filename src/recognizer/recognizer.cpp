#include "offline_asr/recognizer/recognizer.h"

#include "offline_asr/audio/wav_reader.h"
#include "offline_asr/decoder/ctc_wrapper.h"
#include "offline_asr/decoder/token_table.h"
#include "offline_asr/feature/fbank_extractor.h"
#include "offline_asr/runtime/runtime_factory.h"
#include "offline_asr/utils/timer.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace offline_asr {

class Recognizer::Impl {
public:
  explicit Impl(const RecognizerConfig &cfg)
      : cfg_(cfg), fbank_extractor_(FbankExtractor::Config{
                       cfg.sample_rate, cfg.frame_length_ms, cfg.frame_shift_ms,
                       cfg.num_mel_bins, cfg.pre_emphasis_alpha, true}) {}

  bool Init() {
    // Runtime
    runtime_ = RuntimeFactory::Create(cfg_.runtime_backend, cfg_.num_threads);
    if (!runtime_) {
      spdlog::error("Failed to create runtime backend: {}",
                    cfg_.runtime_backend);
      return false;
    }
    if (!runtime_->LoadModel(cfg_.model_path, cfg_.input_name,
                             cfg_.output_name)) {
      return false;
    }

    // Decoder
    CtcDecoderWrapper::Config decoder_cfg;
    decoder_cfg.beam_size = cfg_.beam_size;
    decoder_cfg.n_best = cfg_.n_best;
    decoder_cfg.cutoff_threshold = cfg_.cutoff_threshold;
    decoder_ = std::make_unique<CtcDecoderWrapper>(decoder_cfg);

    // Token table
    if (!cfg_.tokens_path.empty()) {
      if (!token_table_.Load(cfg_.tokens_path)) {
        spdlog::warn("Failed to load tokens from {}, using char vocab",
                     cfg_.tokens_path);
        token_table_ = CreateCharVocab();
      }
    } else {
      token_table_ = CreateCharVocab();
      spdlog::info(
          "No tokens file specified, using default char vocab ({} tokens)",
          token_table_.VocabSize());
    }

    return true;
  }

  RecognitionResult Recognize(const float *samples, size_t num_samples) {
    Timer total_timer, stage_timer;
    RecognitionResult result;

    double audio_duration = static_cast<double>(num_samples) /
                            cfg_.sample_rate * 1000.0; // 化为毫秒
    result.audio_duration_ms = audio_duration;

    // Preprocess: remove DC offset + peak normalize (fused into 2 passes)
    preprocess_buffer_.resize(num_samples);
    float mean = 0.0F;
    for (size_t i = 0; i < num_samples; ++i)
      mean += samples[i];
    mean /= static_cast<float>(num_samples);

    float peak = 0.0F;
    for (size_t i = 0; i < num_samples; ++i) {
      float v = samples[i] - mean;
      preprocess_buffer_[i] = v;
      float abs_v = v < 0.0F ? -v : v;
      if (abs_v > peak)
        peak = abs_v;
    }
    if (peak > 0.0F) {
      float inv_peak = 1.0F / peak;
      for (size_t i = 0; i < num_samples; ++i)
        preprocess_buffer_[i] *= inv_peak;
    }
    result.preprocess_ms = stage_timer.ElapsedMs();

    // Feature extraction
    stage_timer.Reset();
    auto fbank =
        fbank_extractor_.Extract(preprocess_buffer_.data(), num_samples);
    int num_frames = fbank_extractor_.NumFrames(num_samples);
    result.fbank_ms = stage_timer.ElapsedMs();

    if (fbank.empty() || num_frames == 0) {
      spdlog::warn("No FBank features extracted ({} samples)", num_samples);
      return result;
    }

    // Inference
    stage_timer.Reset();
    auto log_probs =
        runtime_->Forward(fbank.data(), num_frames, cfg_.num_mel_bins);
    result.inference_ms = stage_timer.ElapsedMs();

    if (log_probs.empty()) {
      spdlog::error("Inference returned empty result");
      return result;
    }

    int output_frames =
        static_cast<int>(log_probs.size()) / runtime_->VocabSize();
    if (output_frames == 0) {
      spdlog::warn("Zero output frames from model");
      return result;
    }

    // Decode
    stage_timer.Reset();
    auto decode_results =
        decoder_->Decode(log_probs.data(), output_frames,
                         static_cast<int>(runtime_->VocabSize()));
    result.decoder_ms = stage_timer.ElapsedMs();

    if (!decode_results.empty()) {
      result.tokens = decode_results[0].tokens;
      result.confidence = decode_results[0].score;
      result.text = token_table_.Decode(result.tokens);

      // Frame timestamps → ms
      float frame_shift_ms = cfg_.frame_shift_ms * 4.0f; // CNN downsampling 4x
      result.timestamps.reserve(decode_results[0].timestamps.size());
      for (int ts : decode_results[0].timestamps) {
        result.timestamps.push_back(ts * frame_shift_ms);
      }
    }

    double total_ms = total_timer.ElapsedMs();
    result.rtf = total_ms / audio_duration;

    return result;
  }

  RecognitionResult Recognize(const std::string &audio_path) {
    auto audio = WavReader::Read(audio_path);
    if (audio.samples.empty()) {
      spdlog::error("Failed to read audio: {}", audio_path);
      return {};
    }
    // Resample if needed (placeholder - actual resampling TBD)
    if (audio.sample_rate != cfg_.sample_rate) {
      spdlog::warn("Sample rate mismatch: audio={}, expected={}",
                   audio.sample_rate, cfg_.sample_rate);
      // TODO: resampling
    }
    return Recognize(audio.samples.data(), audio.samples.size());
  }

  // Streaming stubs (V1.1)
  void AcceptWaveform(const float *, size_t) {}
  std::string Decode() { return ""; }
  std::string GetPartialResult() { return ""; }
  void Reset() { decoder_->Reset(); }

private:
  RecognizerConfig cfg_;
  FbankExtractor fbank_extractor_;
  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<CtcDecoderWrapper> decoder_;
  TokenTable token_table_;
  std::vector<float> preprocess_buffer_; // reused across calls
};

// ---- PIMPL forwarding ----

Recognizer::Recognizer(const RecognizerConfig &cfg)
    : impl_(std::make_unique<Impl>(cfg)) {
  impl_->Init();
}

Recognizer::Recognizer(const std::string &yaml_path)
    : impl_(std::make_unique<Impl>(RecognizerConfig::FromYaml(yaml_path))) {
  impl_->Init();
}

Recognizer::~Recognizer() = default;

RecognitionResult Recognizer::Recognize(const std::string &audio_path) {
  return impl_->Recognize(audio_path);
}

RecognitionResult Recognizer::Recognize(const float *samples,
                                        size_t num_samples) {
  return impl_->Recognize(samples, num_samples);
}

void Recognizer::AcceptWaveform(const float *samples, size_t num_samples) {
  impl_->AcceptWaveform(samples, num_samples);
}

std::string Recognizer::Decode() { return impl_->Decode(); }
std::string Recognizer::GetPartialResult() { return impl_->GetPartialResult(); }
void Recognizer::Reset() { impl_->Reset(); }

} // namespace offline_asr
