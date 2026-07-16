#include <gtest/gtest.h>

#include "offline_asr/decoder/token_table.h"

using namespace offline_asr;

TEST(TokenTable, CreateCharVocab) {
    auto table = CreateCharVocab();
    EXPECT_EQ(table.VocabSize(), 29);
    EXPECT_EQ(table.BlankId(), 0);
    EXPECT_EQ(table.Encode("<blank>"), 0);
    EXPECT_EQ(table.Encode(" "), 1);
    EXPECT_EQ(table.Encode("a"), 3);
    EXPECT_EQ(table.Encode("z"), 28);
}

TEST(TokenTable, DecodeSimple) {
    auto table = CreateCharVocab();
    // <blank> h e <blank> l l o <blank> -> "helo"
    std::vector<int> ids = {0, 10, 7, 0, 14, 14, 17, 0};
    EXPECT_EQ(table.Decode(ids), "helo");
}

TEST(TokenTable, DecodeWithSpace) {
    auto table = CreateCharVocab();
    // h i <space> t h e r e
    std::vector<int> ids = {10, 11, 1, 22, 10, 7, 20, 7};
    EXPECT_EQ(table.Decode(ids), "hi there");
}

TEST(TokenTable, EmptyInput) {
    auto table = CreateCharVocab();
    EXPECT_EQ(table.Decode(std::vector<int>{}), "");
    EXPECT_EQ(table.Decode(std::vector<int>{0, 0, 0}), "");
}

TEST(TokenTable, LoadFromFile) {
    TokenTable table;
    EXPECT_TRUE(table.Load(TEST_DATA_DIR "/models/tokens.txt"));
    EXPECT_EQ(table.VocabSize(), 29);
    EXPECT_EQ(table.Decode(0), "<blank>");
}

TEST(TokenTable, EncodeNotFound) {
    auto table = CreateCharVocab();
    EXPECT_EQ(table.Encode("XX"), -1);
    EXPECT_EQ(table.Decode(-1), "");
    EXPECT_EQ(table.Decode(999), "");
}
