#include "offline_asr/recognizer/recognizer.h"

#include "offline_asr/audio/wav_reader.h"
#include "offline_asr/decoder/ctc_wrapper.h"
#include "offline_asr/decoder/token_table.h"
#include "offline_asr/feature/fbank_extractor.h"
#include "offline_asr/feature/streaming_fbank_extractor.h"
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

    // Streaming FBank extractor
    StreamingFbankExtractor::Config sfbank_cfg;
    sfbank_cfg.sample_rate = cfg_.sample_rate;
    sfbank_cfg.frame_length_ms = cfg_.frame_length_ms;
    sfbank_cfg.frame_shift_ms = cfg_.frame_shift_ms;
    sfbank_cfg.num_mel_bins = cfg_.num_mel_bins;
    sfbank_cfg.pre_emphasis_alpha = cfg_.pre_emphasis_alpha;
    streaming_fbank_ = std::make_unique<StreamingFbankExtractor>(sfbank_cfg);

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

  // ---- Streaming (V1.1) ----

  void AcceptWaveform(const float *samples, size_t num_samples) {
    if (!streaming_fbank_ || !runtime_ || !decoder_)
      return;

    // Preprocess (same as offline path): DC removal + peak normalize
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

    // Extract new FBank frames from this chunk
    auto new_frames =
        streaming_fbank_->Feed(preprocess_buffer_.data(), num_samples);
    if (new_frames.empty())
      return;

    size_t feat_dim = static_cast<size_t>(streaming_fbank_->NumMelBins());
    chunk_fbank_buffer_.insert(chunk_fbank_buffer_.end(), new_frames.begin(),
                               new_frames.end());

    // Run inference when enough frames accumulated
    int num_mel_bins = streaming_fbank_->NumMelBins();
    while (static_cast<int>(chunk_fbank_buffer_.size() / feat_dim) >=
           kMinChunkFrames) { // 只要缓冲中的帧数超过
                              // kMinChunkFrames，就可以进行推理
      int total_frames =
          static_cast<int>(chunk_fbank_buffer_.size() / feat_dim);

      auto log_probs = runtime_->Forward(chunk_fbank_buffer_.data(),
                                         total_frames, num_mel_bins);
      if (log_probs.empty())
        return;

      int vocab_size = static_cast<int>(runtime_->VocabSize());
      int output_frames = static_cast<int>(log_probs.size()) / vocab_size;

      // Feed per-frame log-probs to CTC decoder
      for (int f = 0; f < output_frames; ++f) {
        decoder_->Step(log_probs.data() + f * vocab_size, vocab_size);
      }

      // Keep overlap frames for next chunk (CNN context smoothing)
      // 为了给下一次推理提供帧间的上下文（尤其当模型包含卷积层时），会保留最后
      // kChunkOverlap 帧
      int keep_frames = total_frames - kChunkOverlap;
      if (keep_frames <= 0) // 如果总帧数不足以保留重叠帧，则清空缓冲区
        keep_frames = total_frames;
      size_t keep_elements = keep_frames * feat_dim;
      chunk_fbank_buffer_.erase(chunk_fbank_buffer_.begin(),
                                chunk_fbank_buffer_.begin() + keep_elements);
    }
  }

  std::string GetPartialResult() {
    if (!decoder_)
      return "";
    auto results = decoder_->Results(cfg_.n_best);
    if (results.empty())
      return "";
    return token_table_.Decode(results[0].tokens);
  }

  std::string Decode() {
    if (!streaming_fbank_ || !runtime_ || !decoder_)
      return "";

    // Flush remaining FBank frames
    auto final_frames = streaming_fbank_->Flush();
    if (!final_frames.empty()) {
      size_t feat_dim = static_cast<size_t>(streaming_fbank_->NumMelBins());
      chunk_fbank_buffer_.insert(chunk_fbank_buffer_.end(),
                                 final_frames.begin(), final_frames.end());

      int total_frames =
          static_cast<int>(chunk_fbank_buffer_.size() / feat_dim);
      if (total_frames > 0) {
        auto log_probs =
            runtime_->Forward(chunk_fbank_buffer_.data(), total_frames,
                              streaming_fbank_->NumMelBins());
        if (!log_probs.empty()) {
          int vocab_size = static_cast<int>(runtime_->VocabSize());
          int output_frames = static_cast<int>(log_probs.size()) / vocab_size;
          for (int f = 0; f < output_frames; ++f) {
            decoder_->Step(log_probs.data() + f * vocab_size, vocab_size);
          }
        }
      }
      chunk_fbank_buffer_.clear();
    }

    auto results = decoder_->Results(1);
    if (results.empty())
      return "";
    return token_table_.Decode(results[0].tokens);
  }

  void Reset() {
    streaming_fbank_->Reset();
    chunk_fbank_buffer_.clear();
    decoder_->Reset();
  }

private:
  RecognizerConfig cfg_;
  FbankExtractor fbank_extractor_;
  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<CtcDecoderWrapper> decoder_;
  TokenTable token_table_;
  std::vector<float> preprocess_buffer_; // reused across calls

  // Streaming state (V1.1)
  std::unique_ptr<StreamingFbankExtractor> streaming_fbank_;
  std::vector<float> chunk_fbank_buffer_;
  static constexpr int kMinChunkFrames = 40;
  static constexpr int kChunkOverlap = 4;
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
