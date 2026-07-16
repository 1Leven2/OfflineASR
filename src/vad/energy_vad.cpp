#include "offline_asr/vad/energy_vad.h"

#include <cmath>

namespace offline_asr {

static int ms_to_frames(int ms, int frame_ms) {
  int f = ms / frame_ms;
  return f > 0 ? f : 1;
}

EnergyVad::EnergyVad(const Config &cfg)
    : cfg_(cfg), frame_size_(cfg.sample_rate * cfg.frame_ms /
                             1000), // 30ms -> 480 samples,
      min_speech_frames_(ms_to_frames(
          cfg.min_speech_ms, cfg.frame_ms)) // 100 / 30 = 3.33 -> 3 frames
      ,
      min_silence_frames_(ms_to_frames(
          cfg.min_silence_ms, cfg.frame_ms)) // 200 / 30 = 6.67 -> 7 frames
{}

float EnergyVad::ComputeRms(const float *samples, size_t n) {
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_sq += static_cast<double>(samples[i]) * samples[i];
  }
  return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
}

bool EnergyVad::IsSpeech(const float *samples, size_t n) {
  if (n == 0)
    return false;

  size_t num_frames = n / static_cast<size_t>(frame_size_);
  bool any_speech = false;

  for (size_t f = 0; f < num_frames; ++f) {
    const float *frame = samples + f * frame_size_;
    float rms = ComputeRms(frame, static_cast<size_t>(frame_size_));
    bool frame_is_speech = (rms > cfg_.threshold);

    // 如果当前帧是语音（能量高）
    if (frame_is_speech) {
      any_speech = true;
      silence_frame_count_ = 0;
      ++speech_frame_count_;

      // 如果当前状态是挂起或静音，则切换到语音状态
      if (state_ == State::kHangover || state_ == State::kSilence) {
        state_ = State::kSpeech;
      }
    } else { // 当前帧是静音（能量低）
      switch (state_) {
      // 如果当前状态是语音，说明之前连续语音帧后遇到了一个低能量帧
      case State::kSpeech:
        // 检查是否已经有足够的连续语音帧，如果是，则进入挂起状态，否则回到静音状态
        if (speech_frame_count_ >= min_speech_frames_) {
          state_ = State::kHangover;
          silence_frame_count_ = 1; // 表示这是挂起状态下的第一帧静音
        } else {
          // 如果不够连续语音帧，说明这段语音太短，可能是噪声或误触发
          // Too short — reject as noise
          state_ = State::kSilence;
          speech_frame_count_ =
              0; // 清零 speech_frame_count_，因为这段语音被认为是无效的
        }
        break;
        // 如果是挂起状态，说明已经在语音结束后的拖延期，继续计数静音帧
        // 但此时又遇到了一个低能量帧，则 silence_frame_count_ 增加
      case State::kHangover:
        ++silence_frame_count_;
        // 如果达到最小静音帧数，则切换到静音状态
        if (silence_frame_count_ >= min_silence_frames_) {
          state_ = State::kSilence;
          speech_frame_count_ =
              0; // 清零 speech_frame_count_，因为挂起状态结束了
          silence_frame_count_ =
              0; // 清零 silence_frame_count_，因为已经切换到静音状态
        }
        break;
      // 如果状态是静音，说明之前已经是静音状态，遇到低能量帧也不需要做任何处理
      case State::kSilence:
        break;
      }
    }
  }

  // 返回当前状态是否为语音或挂起状态，如果是，则认为当前帧是语音
  return (state_ == State::kSpeech || state_ == State::kHangover);
}

void EnergyVad::Reset() {
  state_ = State::kSilence;
  speech_frame_count_ = 0;
  silence_frame_count_ = 0;
}

std::vector<SpeechSegment> Vad::Segment(const float *samples, size_t n) {
  std::vector<SpeechSegment> segments;
  if (n == 0)
    return segments;

  size_t fs = FrameSize();
  if (fs == 0)
    return segments;

  Reset();
  size_t num_frames = n / fs;
  bool in_segment = false;
  size_t seg_start = 0;

  for (size_t f = 0; f < num_frames; ++f) {
    const float *frame = samples + f * fs;
    bool speech = IsSpeech(frame, fs);

    if (speech && !in_segment) {
      in_segment = true;
      seg_start = f * fs;
    } else if (!speech && in_segment) {
      in_segment = false;
      segments.push_back({seg_start, f * fs});
    }
  }

  if (in_segment) {
    segments.push_back({seg_start, n});
  }

  return segments;
}

} // namespace offline_asr
