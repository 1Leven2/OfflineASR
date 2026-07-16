# OfflineASR —— C++17 离线语音识别 Runtime

工业级、可扩展的 C++17 离线语音识别运行时框架。训练交给 PyTorch，推理完全基于 ONNX Runtime，不依赖任何深度学习框架。

```
Audio → FBank → ONNX Runtime → CTC Decoder → Text
```

## 特性

- **纯 C++ 推理**：不依赖 PyTorch/TensorFlow，基于 ONNX Runtime
- **模块化架构**：Audio / Feature / Runtime / Decoder / Recognizer 完全解耦
- **多后端支持**：Runtime 抽象接口，当前支持 ONNX Runtime，未来可扩展 TensorRT / OpenVINO
- **YAML 配置驱动**：无需重新编译即可切换模型和参数
- **Benchmark 内置**：各模块耗时、RTF、内存峰值
- **Streaming 预留**：架构已为流式识别设计接口

## 架构

```
                 OfflineASR Runtime

          ┌─────────────────────────┐
          │      Recognizer         │  ← 唯一对外入口
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
  Audio            Feature           Runtime         Decoder
     │                 │                 │               │
 WavReader           FBank          ONNX Runtime    CTC Decoder
(libsndfile)      (~/FBank/)       (onnxruntime)   (~/CTC/)
```

## 依赖

| 依赖 | 用途 | 备注 |
|------|------|------|
| C++17 | 语言标准 | GCC 9+ / Clang 10+ |
| CMake ≥ 3.18 | 构建系统 | |
| ONNX Runtime | 神经网络推理 | 系统安装或 FetchContent |
| libsndfile | WAV/PCM 读取 | `apt install libsndfile1-dev` |
| spdlog | 日志 | FetchContent 自动下载 |
| yaml-cpp | 配置解析 | FetchContent 自动下载 |
| FBank | 特征提取 | 已有库，链接 `~/FBank/build/libfbank.a` |
| CTC Decoder | CTC 解码 | 已有库，链接 `~/CTC/build/libctc_core.a` |

## 快速开始

### 1. 安装系统依赖

```bash
# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar xzf onnxruntime-linux-x64-1.17.1.tgz
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.17.1

# libsndfile
sudo apt install libsndfile1-dev
```

### 2. 编译

```bash
cd /home/Young/ASR
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)
```

### 3. 准备模型

```bash
# 训练并导出 ONNX 模型
cd python
pip install -r requirements.txt
python train.py          # 在 AN4 上训练 CNN-BiGRU 模型
python export_onnx.py    # 导出为 ONNX 格式
```

### 4. 运行

```bash
# 单文件识别
./build/cli/offline_asr transcribe demo.wav --config configs/default.yaml

# 批量识别
./build/cli/offline_asr batch filelist.txt --output result.txt

# 性能基准
./build/cli/offline_asr benchmark --config configs/default.yaml --iterations 100
```

## CLI 使用

```
offline_asr transcribe <audio.wav> [options]
  --config <path>       YAML 配置文件路径 (默认: configs/default.yaml)

offline_asr batch <filelist.txt> [options]
  --config <path>       YAML 配置文件路径
  --output <path>       结果输出文件
  --threads <int>       线程数

offline_asr benchmark [options]
  --config <path>       YAML 配置文件路径
  --iterations <int>    迭代次数 (默认: 100)
```

## 配置

```yaml
# configs/default.yaml
model:
  path: models/model.onnx       # ONNX 模型路径
  input_name: "input"           # 模型输入节点名
  output_name: "output"         # 模型输出节点名

runtime:
  backend: onnx                 # 推理后端: onnx | tensorrt (未来)
  num_threads: 8                # 推理线程数

feature:
  sample_rate: 16000
  frame_length_ms: 25.0
  frame_shift_ms: 10.0
  num_mel_bins: 80
  pre_emphasis_alpha: 0.97

decoder:
  beam_size: 10                 # CTC 束搜索宽度
  n_best: 1                     # 返回最优结果数
  cutoff_threshold: 0.0         # 剪枝阈值 (0 = 不剪枝)

tokens: models/tokens.txt       # 词汇表文件
```

## tokens.txt 格式

每行一个 token，第一行固定为 `<blank>`：

```
<blank>
<space>
'
a
b
...
z
```

## Benchmark

```
======================
Module       Latency(ms)
----------------------
Audio Read        0.85
FBank             1.42
Inference        12.85
Decoder           0.81
----------------------
Total            15.93

Audio Duration: 1280.00 ms
RTF: 0.012
======================
```

## API 使用

```cpp
#include "offline_asr/recognizer/recognizer.h"

// 从 YAML 配置创建
offline_asr::Recognizer recognizer("configs/default.yaml");

// 识别音频文件
auto result = recognizer.Recognize("demo.wav");
std::cout << result.text << std::endl;       // 识别文本
std::cout << result.rtf << std::endl;        // 实时率

// 从内存识别
std::vector<float> samples = ...;
auto result = recognizer.Recognize(samples.data(), samples.size());
```

## 模型训练

训练流程独立于 Runtime，在 `python/` 下进行：

```bash
cd python
pip install -r requirements.txt

# AN4 数据集训练（数字识别）
python train.py --dataset an4 --data_dir ../data/an4

# LibriSpeech 训练（通用语音识别）
python train.py --dataset librispeech --data_dir ../data/librispeech

# 导出 ONNX
python export_onnx.py --checkpoint checkpoints/best.pt --output ../models/model.onnx

# 验证 ONNX 推理正确性
python infer.py --model ../models/model.onnx --audio ../data/test.wav
```

当前模型架构：

```
CNN (4层 Conv1d + BN + ReLU + MaxPool×2)
  ↓
BiGRU (2层, hidden=256, bidirectional)
  ↓
Linear (512 → 29)
  ↓
LogSoftmax
```

参数量约 3.18M，输入 80 维 FBank，输出 29 类字符级 log 概率。

## 测试

```bash
cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

## 项目结构

```
/home/Young/ASR/
├── CMakeLists.txt
├── README.md
├── cmake/                        # CMake 模块 (FindONNXRuntime 等)
├── include/offline_asr/          # 公共头文件
│   ├── audio/                    #   音频读取
│   ├── feature/                  #   特征提取 (封装 FBank)
│   ├── runtime/                  #   推理后端 (ONNX Runtime)
│   ├── decoder/                  #   CTC 解码 (封装 CTC)
│   ├── recognizer/               #   顶层 API + 配置
│   └── utils/                    #   日志、计时器、线程池
├── src/                          # 实现文件
├── cli/                          # 命令行工具
├── python/                       # 训练 + ONNX 导出
├── models/                       # ONNX 模型 + tokens.txt
├── configs/                      # YAML 配置文件
├── tests/                        # GoogleTest 单元测试
├── benchmark/                    # Google Benchmark
└── docs/                         # 文档
```

## 路线图

| 版本 | 内容 |
|------|------|
| V1.0 | 离线识别 pipeline, ONNX Runtime, CLI, Benchmark |
| V1.1 | Streaming 识别, 部分结果, Profiler |
| V1.2 | Silero VAD, YAML Config 完善, Logger |
| V2.0 | TensorRT Runtime, Batch 推理, Async Pipeline |
| V3.0 | Conformer / Paraformer / Whisper Encoder 多模型支持 |
