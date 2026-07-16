#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace offline_asr {

/**
 * Token 映射表 — 管理 token ID ↔ 文本 的双向映射
 *
 * 约定: token 0 固定为 <blank> (CTC blank token)
 */
class TokenTable {
public:
    TokenTable() = default;

    /** 从文件加载 (每行一个 token，首行为 <blank>) */
    bool Load(const std::string& path);

    /** 从 token 列表构建 */
    explicit TokenTable(const std::vector<std::string>& tokens);

    int Encode(const std::string& token) const;
    std::string Decode(int id) const;

    /** CTC collapse: 去 blank + 合并相邻重复后解码为文本 */
    std::string Decode(const std::vector<int>& ids) const;

    int VocabSize() const noexcept { return static_cast<int>(id_to_token_.size()); }
    int BlankId() const noexcept { return 0; }

    bool Empty() const noexcept { return id_to_token_.empty(); }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
};

/** 返回标准 29 字符词汇表: <blank> + <space> + ' + a-z */
TokenTable CreateCharVocab();

}  // namespace offline_asr
