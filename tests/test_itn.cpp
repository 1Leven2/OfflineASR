#include <gtest/gtest.h>

#include "offline_asr/recognizer/itn.h"

using namespace offline_asr;

TEST(ITN, Cardinal) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("一万两千三百四十五"), "12345");
    EXPECT_EQ(itn.Process("一百"), "100");
    EXPECT_EQ(itn.Process("一千零一"), "1001");
}

TEST(ITN, Ordinal) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("第三十五"), "第35");
    EXPECT_EQ(itn.Process("第一"), "第1");
    EXPECT_EQ(itn.Process("第十二"), "第12");
}

TEST(ITN, Money) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("一千二百三十四元"), "1234元");
    EXPECT_EQ(itn.Process("五元三角"), "5.30元");
    EXPECT_EQ(itn.Process("十元五角六分"), "10.56元");
}

TEST(ITN, Percent) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("百分之五十"), "50%");
    EXPECT_EQ(itn.Process("百分之九十九点五"), "99.5%");
    EXPECT_EQ(itn.Process("百分之百"), "100%"); // 百分之百 = 百分之(一)百 = 100%
}

TEST(ITN, Date) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("二零零八年一月二十八日"), "2008年1月28日");
    EXPECT_EQ(itn.Process("二零二零年十二月一日"), "2020年12月1日");
}

TEST(ITN, Time) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("三点二十分"), "03:20");
    EXPECT_EQ(itn.Process("十二点零五分"), "12:05");
}

TEST(ITN, Decimal) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("三点一四"), "3.14");
    EXPECT_EQ(itn.Process("十二点五"), "12.5");
}

TEST(ITN, MixedText) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("价格是一千二百元"), "价格是1200元");
    EXPECT_EQ(itn.Process("今天是二零零八年一月二十八日"), "今天是2008年1月28日");
}

TEST(ITN, NoNumber) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("今天天气很好"), "今天天气很好");
    EXPECT_EQ(itn.Process("hello world"), "hello world");
}

TEST(ITN, Empty) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process(""), "");
}

TEST(ITN, Disabled) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = false;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("一万两千三百四十五"), "一万两千三百四十五");
    EXPECT_EQ(itn.Process("第三十五"), "第三十五");
}

TEST(ITN, ComplexDate) {
    InverseTextNormalizer::Config cfg;
    cfg.enabled = true;
    InverseTextNormalizer itn(cfg);

    EXPECT_EQ(itn.Process("一九九九年十二月三十一日"), "1999年12月31日");
}
