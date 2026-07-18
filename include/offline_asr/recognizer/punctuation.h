#pragma once

#include <string>
#include <vector>

#include "offline_asr/decoder/token_table.h"

namespace offline_asr {

class PunctuationInserter {
public:
    struct Config {
        bool enabled = false;
        int period_silence_ms = 800;
        int comma_silence_ms = 300;
    };

    explicit PunctuationInserter(const Config& cfg);

    /**
     * Insert punctuation into decoded text based on inter-word silence gaps.
     * @param tokens  raw token IDs (before CTC collapse)
     * @param timestamps_ms  per-token timestamps in ms (after CNN subsampling)
     * @param token_table  token→string lookup
     * @param is_final  if true, append period at end
     */
    std::string Process(const std::vector<int>& tokens,
                        const std::vector<float>& timestamps_ms,
                        const TokenTable& token_table,
                        bool is_final);

private:
    struct Word {
        std::string text;
        float start_ms = 0.0f;
        float end_ms = 0.0f;
    };

    std::vector<Word> ExtractWords(const std::vector<int>& tokens,
                                   const std::vector<float>& timestamps_ms,
                                   const TokenTable& token_table);

    Config cfg_;
};

}  // namespace offline_asr
