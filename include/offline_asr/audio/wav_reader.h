#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace offline_asr {

/**
 * 音频读取器 — 基于 libsndfile 的 WAV/PCM 文件读取
 *
 * 自动处理:
 *   - 多声道 → 单声道混音
 *   - 整型采样 → 浮点归一化 [-1, 1]
 */
class WavReader {
public:
    struct AudioData {
        std::vector<float> samples;  // 归一化到 [-1, 1]
        int32_t sample_rate = 0;
        int32_t num_channels = 0;
    };

    /** 从文件路径读取 */
    static AudioData Read(const std::string& path);

    /** 从内存读取 */
    static AudioData Read(const void* data, size_t size);
};

}  // namespace offline_asr
