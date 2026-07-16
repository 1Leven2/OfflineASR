#include "offline_asr/decoder/ctc_wrapper.h"

#include <ctc/prefix_beam_search.h>
#include <spdlog/spdlog.h>

namespace offline_asr {

class CtcDecoderWrapper::Impl {
public:
    explicit Impl(const Config& cfg) {
        opts_.beam_size = cfg.beam_size;
        opts_.blank_id = cfg.blank_id;
        opts_.n_best = cfg.n_best;
        opts_.cutoff_threshold = cfg.cutoff_threshold;
        decoder_ = std::make_unique<ctc::PrefixBeamSearch>(opts_);
    }

    std::vector<Result> Decode(const float* log_probs, int num_frames, int vocab_size) {
        auto results = decoder_->Decode(log_probs, num_frames, vocab_size);
        std::vector<Result> out;
        out.reserve(results.size());
        for (auto& r : results) {
            out.push_back({std::move(r.tokens), r.score, std::move(r.timestamps)});
        }
        return out;
    }

    void Step(const float* log_probs, int vocab_size) {
        decoder_->Step(log_probs, vocab_size);
    }

    std::vector<Result> Results(int n) const {
        auto results = decoder_->Results(n);
        std::vector<Result> out;
        out.reserve(results.size());
        for (auto& r : results) {
            // timestamps from const Results() are not available, use empty
            out.push_back({r.tokens, r.score, {}});
        }
        return out;
    }

    void Reset() {
        decoder_->Reset();
    }

private:
    ctc::DecoderOptions opts_;
    std::unique_ptr<ctc::PrefixBeamSearch> decoder_;
};

CtcDecoderWrapper::CtcDecoderWrapper(const Config& cfg)
    : cfg_(cfg)
    , impl_(std::make_unique<Impl>(cfg)) {}

CtcDecoderWrapper::~CtcDecoderWrapper() = default;

std::vector<CtcDecoderWrapper::Result> CtcDecoderWrapper::Decode(
    const float* log_probs, int num_frames, int vocab_size) {
    return impl_->Decode(log_probs, num_frames, vocab_size);
}

void CtcDecoderWrapper::Step(const float* log_probs, int vocab_size) {
    impl_->Step(log_probs, vocab_size);
}

std::vector<CtcDecoderWrapper::Result> CtcDecoderWrapper::Results(int n) const {
    return impl_->Results(n);
}

void CtcDecoderWrapper::Reset() {
    impl_->Reset();
}

}  // namespace offline_asr
