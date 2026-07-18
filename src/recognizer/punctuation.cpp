#include "offline_asr/recognizer/punctuation.h"

namespace offline_asr {

PunctuationInserter::PunctuationInserter(const Config& cfg) : cfg_(cfg) {}

std::vector<PunctuationInserter::Word>
PunctuationInserter::ExtractWords(const std::vector<int>& tokens,
                                  const std::vector<float>& timestamps_ms,
                                  const TokenTable& token_table) {
    std::vector<Word> words;
    Word current;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string ch = token_table.Decode(tokens[i]);
        if (ch == " ") {
            if (!current.text.empty()) {
                current.end_ms = timestamps_ms[i - 1];
                words.push_back(current);
                current = Word{};
            }
        } else {
            if (current.text.empty())
                current.start_ms = timestamps_ms[i];
            current.text += ch;
        }
    }
    if (!current.text.empty()) {
        current.end_ms = timestamps_ms.back();
        words.push_back(current);
    }
    return words;
}

std::string PunctuationInserter::Process(const std::vector<int>& tokens,
                                         const std::vector<float>& timestamps_ms,
                                         const TokenTable& token_table,
                                         bool is_final) {
    if (!cfg_.enabled)
        return token_table.Decode(tokens);

    auto words = ExtractWords(tokens, timestamps_ms, token_table);
    if (words.empty())
        return "";

    std::string result;
    for (size_t i = 0; i < words.size(); ++i) {
        result += words[i].text;
        if (i + 1 < words.size()) {
            float gap = words[i + 1].start_ms - words[i].end_ms;
            if (gap > static_cast<float>(cfg_.period_silence_ms))
                result += "\xe3\x80\x82";  // "。"
            else if (gap > static_cast<float>(cfg_.comma_silence_ms))
                result += "\xef\xbc\x8c";  // "，"
            else
                result += " ";  // normal word boundary
        }
    }

    if (is_final && !result.empty()) {
        unsigned char last = static_cast<unsigned char>(result.back());
        // Skip if already ends with punctuation
        bool has_punct = false;
        if (result.size() >= 3) {
            std::string tail = result.substr(result.size() - 3);
            if (tail == "\xe3\x80\x82" || tail == "\xef\xbc\x8c")  // 。or ，
                has_punct = true;
        }
        if (!has_punct && last != '.' && last != ',' && last != '!' && last != '?')
            result += "\xe3\x80\x82";  // "。"
    }

    return result;
}

}  // namespace offline_asr
