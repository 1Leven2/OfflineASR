#include "offline_asr/decoder/token_table.h"

#include <spdlog/spdlog.h>
#include <fstream>

namespace offline_asr {

bool TokenTable::Load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open token file: {}", path);
        return false;
    }

    id_to_token_.clear();
    token_to_id_.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        int id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(line);
        token_to_id_[line] = id;
    }

    spdlog::info("Loaded {} tokens from {}", id_to_token_.size(), path);
    return !id_to_token_.empty();
}

TokenTable::TokenTable(const std::vector<std::string>& tokens) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        id_to_token_.push_back(tokens[i]);
        token_to_id_[tokens[i]] = static_cast<int>(i);
    }
}

int TokenTable::Encode(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
}

std::string TokenTable::Decode(int id) const {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[id];
    }
    return "";
}

std::string TokenTable::Decode(const std::vector<int>& ids) const {
    std::string result;
    int prev = -1;
    for (int id : ids) {
        if (id == BlankId()) {
            prev = -1;
            continue;
        }
        if (id == prev) continue;  // CTC merge
        result += Decode(id);
        prev = id;
    }
    return result;
}

TokenTable CreateCharVocab() {
    std::vector<std::string> tokens = {"<blank>", " "};
    tokens.push_back("'");
    for (char c = 'a'; c <= 'z'; ++c) {
        tokens.push_back(std::string(1, c));
    }
    return TokenTable(tokens);
}

}  // namespace offline_asr
