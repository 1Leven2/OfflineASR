#include "offline_asr/feature/streaming_fbank_extractor.h"

#include <cmath>
#include <cstring>

#include <fbank/fft.h>
#include <fbank/log_compression.h>
#include <fbank/mel_filter.h>
#include <fbank/window.h>

namespace offline_asr {

StreamingFbankExtractor::StreamingFbankExtractor(const Config &cfg)
    : num_mel_bins_(cfg.num_mel_bins), alpha_(cfg.pre_emphasis_alpha),
      log_floor_(cfg.log_floor) {

  frame_length_ =
      static_cast<int>(cfg.frame_length_ms * cfg.sample_rate / 1000.0f);
  frame_shift_ =
      static_cast<int>(cfg.frame_shift_ms * cfg.sample_rate / 1000.0f);
  n_fft_ = fbank::next_pow2(frame_length_);
  num_fft_bins_ = n_fft_ / 2 + 1;

  // Pre-compute window (Povey = Hann^0.85, Kaldi default)
  window_ = fbank::make_window(frame_length_, fbank::WindowType::Povey);

  // Pre-compute mel filterbank matrix
  float high_freq =
      cfg.high_freq > 0.0f ? cfg.high_freq : cfg.sample_rate / 2.0f;
  mel_matrix_ = fbank::mel_filterbank(num_mel_bins_, num_fft_bins_,
                                      cfg.sample_rate, cfg.low_freq, high_freq);
}

std::vector<float> StreamingFbankExtractor::Feed(const float *samples,
                                                 size_t n) {
  std::vector<float> new_frames;

  total_fed_ += n;

  for (size_t i = 0; i < n; ++i) {
    // Stateful pre-emphasis: y = x[n] - alpha * x[n-1]
    float x = samples[i];
    float y = x - alpha_ * prev_sample_;
    prev_sample_ = x;
    buffer_.push_back(y);

    // Emit a frame when buffer has enough samples
    while (static_cast<int>(buffer_.size()) >= frame_length_) {
      auto feat = ProcessFrame(buffer_.data());
      new_frames.insert(new_frames.end(), feat.begin(), feat.end());
      ++frames_emitted_;

      // Advance by frame_shift (drop oldest frame_shift samples)
      buffer_.erase(buffer_.begin(), buffer_.begin() + frame_shift_);
    }
  }

  return new_frames;
}

std::vector<float> StreamingFbankExtractor::Flush() {
  if (buffer_.empty())
    return {};

  // Only flush if the buffer contains new samples beyond the last emitted
  // frame's coverage. If no frames were emitted, all content is new.
  // Require at least frame_shift_ new samples to avoid near-zero frames.
  // 让 Flush() 只在有新内容时才产出帧
  if (frames_emitted_ > 0) {
    int last_coverage_end =
        (frames_emitted_ - 1) * frame_shift_ + frame_length_;
    if (static_cast<int>(total_fed_) - last_coverage_end <
        frame_shift_) { // 采样点必须超过上次发射帧的覆盖范围
      buffer_.clear();
      return {};
    }
  }

  // Zero-pad to frame_length
  buffer_.resize(frame_length_, 0.0f);
  auto feat = ProcessFrame(buffer_.data());
  ++frames_emitted_;
  buffer_.clear();
  return feat;
}

void StreamingFbankExtractor::Reset() {
  buffer_.clear();
  prev_sample_ = 0.0f;
  total_fed_ = 0;
  frames_emitted_ = 0;
}

std::vector<float>
StreamingFbankExtractor::ProcessFrame(const float *frame) const {
  // Copy frame for in-place windowing
  std::vector<float> windowed(frame, frame + frame_length_);
  fbank::apply_window(windowed, frame_length_, window_);

  // FFT → power spectrum (single frame)
  auto power_spec =
      fbank::compute_power_spectrum(windowed.data(), 1, frame_length_, n_fft_);

  // Mel filterbank (single frame)
  auto mel = fbank::apply_mel_filterbank(power_spec.data(), 1, num_fft_bins_,
                                         mel_matrix_.data(), num_mel_bins_);

  // Log compression
  fbank::log_compression(mel, log_floor_);

  return mel; // [num_mel_bins]
}

} // namespace offline_asr
