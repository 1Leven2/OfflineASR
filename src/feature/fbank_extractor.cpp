#include "offline_asr/feature/fbank_extractor.h"

#include <fbank/fbank.h>

namespace offline_asr {

FbankExtractor::FbankExtractor(const Config& cfg) : cfg_(cfg) {}

std::vector<float> FbankExtractor::Extract(const float* samples, size_t num_samples) {
    fbank::FbankOptions opts;
    opts.sample_rate = cfg_.sample_rate;
    opts.frame_length_ms = cfg_.frame_length_ms;
    opts.frame_shift_ms = cfg_.frame_shift_ms;
    opts.num_mel_bins = cfg_.num_mel_bins;
    opts.pre_emphasis_alpha = cfg_.pre_emphasis_alpha;
    opts.use_log = cfg_.use_log;
    return fbank::compute_fbank(samples, num_samples, opts);
}

int32_t FbankExtractor::NumFrames(size_t num_samples) const {
    fbank::FbankOptions opts;
    opts.sample_rate = cfg_.sample_rate;
    opts.frame_length_ms = cfg_.frame_length_ms;
    opts.frame_shift_ms = cfg_.frame_shift_ms;
    return fbank::fbank_num_frames(num_samples, opts);
}

}  // namespace offline_asr
