#pragma once

#include <functional>
#include <regex>
#include <string>
#include <vector>

namespace offline_asr {

class InverseTextNormalizer {
public:
    struct Config {
        bool enabled = false;
    };

    explicit InverseTextNormalizer(const Config& cfg);

    std::string Process(const std::string& text);

private:
    using Rule = std::pair<std::regex, std::function<std::string(const std::smatch&)>>;

    Config cfg_;
    std::vector<Rule> rules_;
};

}  // namespace offline_asr
