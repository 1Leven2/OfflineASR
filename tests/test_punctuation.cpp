#include <gtest/gtest.h>

#include "offline_asr/decoder/token_table.h"
#include "offline_asr/recognizer/punctuation.h"

using namespace offline_asr;

namespace {

// Build vocab: <blank>, <space>, a, b, c, d, e, h, i, l, o, r, w, y
// IDs:           0        1        2  3  4  5  6  7  8  9  10 11 12 13
TokenTable MakeVocab() {
    return TokenTable({"<blank>", " ", "a", "b", "c", "d", "e",
                       "h", "i", "l", "o", "r", "w", "y"});
}

}  // namespace

// Helper: given vocab above, "hi" = {7,8}, "bye" = {3,13,6}

TEST(PunctuationInserter, PeriodByGap) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    cfg.period_silence_ms = 400;
    cfg.comma_silence_ms = 200;
    PunctuationInserter punct(cfg);

    // "hi bye": h(7), i(8), space(1), b(3), y(13), e(6)
    std::vector<int> tokens = {7, 8, 1, 3, 13, 6};
    // hi ends at 100ms, bye starts at 580ms → gap = 480ms > 400 period threshold
    std::vector<float> ts = {0, 100, 110, 580, 590, 600};

    std::string result = punct.Process(tokens, ts, vocab, false);
    EXPECT_EQ(result, "hi\xe3\x80\x82"
                      "bye");  // hi。bye
}

TEST(PunctuationInserter, CommaByGap) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    cfg.period_silence_ms = 800;
    cfg.comma_silence_ms = 200;
    PunctuationInserter punct(cfg);

    std::vector<int> tokens = {7, 8, 1, 3, 13, 6};
    // hi ends at 100ms, bye starts at 400ms → gap = 400-100=300ms > 200 comma
    std::vector<float> ts = {0, 100, 110, 400, 410, 420};

    std::string result = punct.Process(tokens, ts, vocab, false);
    EXPECT_EQ(result, "hi\xef\xbc\x8c"
                      "bye");  // hi，bye
}

TEST(PunctuationInserter, NoPunctuationShortGap) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    cfg.period_silence_ms = 800;
    cfg.comma_silence_ms = 300;
    PunctuationInserter punct(cfg);

    std::vector<int> tokens = {7, 8, 1, 3, 13, 6};
    // hi ends at 100ms, bye starts at 150ms → gap = 50ms — below both
    std::vector<float> ts = {0, 100, 110, 150, 160, 170};

    std::string result = punct.Process(tokens, ts, vocab, false);
    EXPECT_EQ(result, "hi bye");
}

TEST(PunctuationInserter, FinalPeriod) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    PunctuationInserter punct(cfg);

    // Just "hi"
    std::vector<int> tokens = {7, 8};
    std::vector<float> ts = {0, 100};

    std::string result = punct.Process(tokens, ts, vocab, true);
    EXPECT_EQ(result, "hi\xe3\x80\x82");  // "hi。"
}

TEST(PunctuationInserter, EmptyTokens) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    PunctuationInserter punct(cfg);

    std::string result = punct.Process({}, {}, vocab, false);
    EXPECT_EQ(result, "");
}

TEST(PunctuationInserter, SingleWord) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    PunctuationInserter punct(cfg);

    // Just "bye"
    std::vector<int> tokens = {3, 13, 6};
    std::vector<float> ts = {0, 10, 20};

    std::string result = punct.Process(tokens, ts, vocab, false);
    EXPECT_EQ(result, "bye");
}

TEST(PunctuationInserter, Disabled) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = false;
    PunctuationInserter punct(cfg);

    std::vector<int> tokens = {7, 8, 1, 3, 13, 6};  // "hi bye"
    std::vector<float> ts = {0, 100, 110, 580, 590, 600};

    // When disabled, just CTC-decoded text
    std::string result = punct.Process(tokens, ts, vocab, false);
    EXPECT_EQ(result, "hi bye");
}

TEST(PunctuationInserter, LongGapMultipleSentences) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    cfg.period_silence_ms = 400;
    cfg.comma_silence_ms = 200;
    PunctuationInserter punct(cfg);

    // "a bye hi" with varying gaps: a(2), space(1), b(3), y(13), e(6), space(1), h(7), i(8)
    std::vector<int> tokens = {2, 1, 3, 13, 6, 1, 7, 8};
    // a ends at 50, bye starts at 500 → gap 450 > 400 → 。
    // bye ends at 540, hi starts at 780 → gap 240 > 200 → ，
    std::vector<float> ts = {0, 50, 500, 510, 540, 550, 780, 790};

    std::string result = punct.Process(tokens, ts, vocab, false);
    EXPECT_EQ(result, "a\xe3\x80\x82"
                      "bye\xef\xbc\x8c"
                      "hi");  // a。bye，hi
}

TEST(PunctuationInserter, TimestampPropagation) {
    auto vocab = MakeVocab();
    PunctuationInserter::Config cfg;
    cfg.enabled = true;
    cfg.period_silence_ms = 400;
    PunctuationInserter punct(cfg);

    // "ab" + space + "cd"
    // Note: CTC collapses "ab" (consecutive a,b) — actually a(2), b(3) are not consecutive identical
    std::vector<int> tokens = {2, 3, 1, 4, 5};
    // "ab" ends at 120ms (timestamp of b), "cd" starts at 600ms (timestamp of c)
    std::vector<float> ts = {100, 120, 130, 600, 610};

    std::string result = punct.Process(tokens, ts, vocab, false);
    // Gap: 600-120=480ms > 400 → 。
    EXPECT_EQ(result, "ab\xe3\x80\x82"
                      "cd");  // ab。cd
}
