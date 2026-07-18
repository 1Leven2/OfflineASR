#include "offline_asr/recognizer/itn.h"

#include <unordered_map>

namespace offline_asr {

namespace {

// Parse digit-by-digit (for years, IDs, etc.): 二零零八 → 2008
int64_t ParseChineseDigits(const std::string& s) {
    static const std::unordered_map<std::string, int64_t> digits = {
        {"零", 0}, {"一", 1}, {"二", 2}, {"三", 3}, {"四", 4},
        {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9}
    };
    int64_t result = 0;
    for (size_t i = 0; i < s.size(); ) {
        int clen = 1;
        unsigned char c = s[i];
        if ((c & 0xF0) == 0xF0) clen = 4;
        else if ((c & 0xE0) == 0xE0) clen = 3;
        else if ((c & 0xC0) == 0xC0) clen = 2;
        std::string ch = s.substr(i, clen);
        i += clen;
        auto it = digits.find(ch);
        if (it != digits.end())
            result = result * 10 + it->second;
    }
    return result;
}

int64_t ParseChineseNumber(const std::string& s) {
    static const std::unordered_map<std::string, int64_t> digits = {
        {"零", 0}, {"一", 1}, {"二", 2}, {"两", 2}, {"三", 3}, {"四", 4},
        {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9}
    };

    if (s.empty()) return 0;

    int64_t result = 0;       // final result
    int64_t segment = 0;      // current segment (below 万)
    int64_t current = 0;      // current number before next unit
    bool has_current = false; // whether we have a pending digit

    for (size_t i = 0; i < s.size(); ) {
        // Try to match a multi-byte character
        int char_len = 1;
        if ((s[i] & 0x80) && (s[i] & 0x40)) {
            // Count continuation bytes
            unsigned char c = s[i];
            if ((c & 0xE0) == 0xC0) char_len = 1; // actually 2, but let me simplify
            // Use UTF-8 encoding: leading byte determines length
            if ((c & 0xF0) == 0xF0) char_len = 4;
            else if ((c & 0xE0) == 0xE0) char_len = 3;
            else if ((c & 0xC0) == 0xC0) char_len = 2;
        }

        std::string ch = s.substr(i, char_len);
        i += char_len;

        if (ch == "十") {
            int64_t val = has_current ? current : 1;
            segment += val * 10;
            has_current = false;
            current = 0;
        } else if (ch == "百") {
            segment += (has_current ? current : 1) * 100;
            has_current = false;
            current = 0;
        } else if (ch == "千") {
            segment += (has_current ? current : 1) * 1000;
            has_current = false;
            current = 0;
        } else if (ch == "万") {
            if (has_current) segment += current;
            result += (segment > 0 ? segment : 1) * 10000;
            segment = 0;
            has_current = false;
            current = 0;
        } else if (ch == "亿") {
            if (has_current) segment += current;
            result += segment;
            result *= 100000000;
            segment = 0;
            has_current = false;
            current = 0;
        } else {
            // Digit
            auto it = digits.find(ch);
            if (it != digits.end()) {
                if (has_current) {
                    // Consecutive digits: "二零" in year
                    segment += current;
                }
                current = it->second;
                has_current = true;
            }
        }
    }

    if (has_current) segment += current;
    result += segment;

    return result;
}

// Format integer removing leading zeros from year-like parts but keeping
// natural decimal notation
std::string FormatInt(int64_t n) {
    return std::to_string(n);
}

}  // namespace

InverseTextNormalizer::InverseTextNormalizer(const Config& cfg) : cfg_(cfg) {
    // Rules are applied in order. Each rule matches a pattern and replaces.

    // 1. Percentage: 百分之九十九点五 → 99.5%
    rules_.emplace_back(
        std::regex(R"(百分之([零一二三四五六七八九两十百千万亿点]+))"),
        [](const std::smatch& m) -> std::string {
            std::string num_part = m[1].str();
            // Check for decimal point
            size_t dot_pos = num_part.find("点");
            if (dot_pos != std::string::npos) {
                std::string int_part = num_part.substr(0, dot_pos);
                std::string frac_part = num_part.substr(dot_pos + 3); // skip "点" (3 bytes)
                int64_t int_val = ParseChineseNumber(int_part);
                std::string result = FormatInt(int_val) + ".";
                for (size_t i = 0; i < frac_part.size(); ) {
                    int clen = 1;
                    unsigned char c = frac_part[i];
                    if ((c & 0xF0) == 0xF0) clen = 4;
                    else if ((c & 0xE0) == 0xE0) clen = 3;
                    else if ((c & 0xC0) == 0xC0) clen = 2;
                    std::string ch = frac_part.substr(i, clen);
                    i += clen;
                    static const std::unordered_map<std::string, int> digits = {
                        {"零", 0}, {"一", 1}, {"二", 2}, {"三", 3}, {"四", 4},
                        {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9}
                    };
                    auto it = digits.find(ch);
                    if (it != digits.end()) result += std::to_string(it->second);
                }
                return result + "%";
            }
            return FormatInt(ParseChineseNumber(num_part)) + "%";
        });

    // 2. Ordinal: 第三十五 → 第35
    rules_.emplace_back(
        std::regex(R"(第([零一二三四五六七八九两十百千万亿]+))"),
        [](const std::smatch& m) -> std::string {
            return "第" + FormatInt(ParseChineseNumber(m[1].str()));
        });

    // 3. Money: number元 (number角) (number分)
    rules_.emplace_back(
        std::regex(R"(([零一二三四五六七八九两十百千万亿]+)元(?:([零一二三四五六七八九两十百千万亿]+)角)?(?:([零一二三四五六七八九两十百千万亿]+)分)?)"),
        [](const std::smatch& m) -> std::string {
            std::string result = FormatInt(ParseChineseNumber(m[1].str()));
            bool has_jiao = m[2].matched && !m[2].str().empty();
            bool has_fen = m[3].matched && !m[3].str().empty();
            if (has_jiao || has_fen) {
                result += ".";
                result += has_jiao ? std::to_string(ParseChineseNumber(m[2].str())) : "0";
                result += has_fen ? std::to_string(ParseChineseNumber(m[3].str())) : "0";
            }
            return result + "元";
        });

    // 4. Date: 二零零八年一月二十八日 → 2008年1月28日
    rules_.emplace_back(
        std::regex(R"(([零一二三四五六七八九两十百千万亿]+)年([零一二三四五六七八九两十百千万亿]+)月([零一二三四五六七八九两十百千万亿]+)日)"),
        [](const std::smatch& m) -> std::string {
            return FormatInt(ParseChineseDigits(m[1].str())) + "年" +
                   FormatInt(ParseChineseNumber(m[2].str())) + "月" +
                   FormatInt(ParseChineseNumber(m[3].str())) + "日";
        });

    // 5. Time: 下午三点二十分 → 15:20  (simplified: just handle "X点Y分")
    rules_.emplace_back(
        std::regex(R"(([零一二三四五六七八九两十百千万亿]+)点([零一二三四五六七八九两十百千万亿]+)分)"),
        [](const std::smatch& m) -> std::string {
            auto h = ParseChineseNumber(m[1].str());
            auto min = ParseChineseNumber(m[2].str());
            char buf[16];
            snprintf(buf, sizeof(buf), "%02d:%02d", static_cast<int>(h), static_cast<int>(min));
            return std::string(buf);
        });

    // 6. Decimal: 三点一四 → 3.14
    rules_.emplace_back(
        std::regex(R"(([零一二三四五六七八九两十百千万亿]+)点([零一二三四五六七八九]+))"),
        [](const std::smatch& m) -> std::string {
            std::string result = FormatInt(ParseChineseNumber(m[1].str())) + ".";
            std::string frac_part = m[2].str();
            for (size_t i = 0; i < frac_part.size(); ) {
                int clen = 1;
                unsigned char c = frac_part[i];
                if ((c & 0xF0) == 0xF0) clen = 4;
                else if ((c & 0xE0) == 0xE0) clen = 3;
                else if ((c & 0xC0) == 0xC0) clen = 2;
                std::string ch = frac_part.substr(i, clen);
                i += clen;
                static const std::unordered_map<std::string, int> digits = {
                    {"零", 0}, {"一", 1}, {"二", 2}, {"三", 3}, {"四", 4},
                    {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9}
                };
                auto it = digits.find(ch);
                if (it != digits.end()) result += std::to_string(it->second);
            }
            return result;
        });

    // 7. Cardinal (pure numbers, must be last to not break other rules): >= 4 digits
    rules_.emplace_back(
        std::regex(R"(([零一二三四五六七八九两十百千万亿]{4,}))"),
        [](const std::smatch& m) -> std::string {
            return FormatInt(ParseChineseNumber(m[1].str()));
        });
}

std::string InverseTextNormalizer::Process(const std::string& text) {
    if (!cfg_.enabled) return text;

    std::string result = text;
    for (const auto& rule : rules_) {
        std::string tmp;
        auto begin = std::sregex_iterator(result.begin(), result.end(), rule.first);
        auto end = std::sregex_iterator();
        size_t last = 0;
        for (auto it = begin; it != end; ++it) {
            tmp += result.substr(last, it->position() - last);
            tmp += rule.second(*it);
            last = it->position() + it->length();
        }
        tmp += result.substr(last);
        result = std::move(tmp);
    }
    return result;
}

}  // namespace offline_asr
