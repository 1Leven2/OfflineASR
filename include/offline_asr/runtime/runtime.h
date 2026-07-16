#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace offline_asr {

/**
 * 推理后端抽象接口
 *
 * 所有推理后端 (ONNX, TensorRT, OpenVINO) 均实现此接口。
 * Recognizer 只依赖此接口，不感知具体后端。
 */
class Runtime {
public:
    virtual ~Runtime() = default;

    /**
     * 加载模型
     * @param path        模型文件路径
     * @param input_name  输入节点名
     * @param output_name 输出节点名
     * @return 是否加载成功
     */
    virtual bool LoadModel(const std::string& path,
                           const std::string& input_name,
                           const std::string& output_name) = 0;

    /**
     * 前向推理
     * @param input      输入数据 (row-major, shape [num_frames, feat_dim])
     * @param num_frames 帧数
     * @param feat_dim   特征维度
     * @return 输出数据 (row-major, shape [num_frames_out, vocab_size])
     */
    virtual std::vector<float> Forward(const float* input,
                                       int64_t num_frames,
                                       int64_t feat_dim) = 0;

    /** 获取输出帧数 */
    virtual int64_t NumOutputFrames(int64_t num_input_frames) const = 0;
    virtual int64_t VocabSize() const = 0;

    virtual std::string BackendName() const = 0;
};

}  // namespace offline_asr
