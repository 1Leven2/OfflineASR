# OfflineASR —— 工业级 C++17 离线语音识别 Runtime 项目计划

> **目标**
>
> 构建一个工业级、可扩展、纯 C++17 的 Offline ASR Runtime。
>
> **推理全流程采用：**
>
> ```
> Audio → FBank → ONNX Runtime → CTC Decoder → Text
> ```
>
> 训练完全交给 PyTorch，Runtime 不依赖任何深度学习框架。

---

# 项目目标

本项目**不是重新实现深度学习框架**，而是实现一个真正可用于部署的 **Speech Runtime**。

最终希望达到：

- 工业级代码组织
- 工业级模块解耦
- 支持多 Runtime Backend
- 支持 Streaming
- 支持 Benchmark
- 支持未来扩展 TensorRT/OpenVINO
- 可作为个人代表项目长期维护

---

# 技术路线

```
                   OfflineASR Runtime

            ┌─────────────────────────┐
            │      Recognizer         │
            └────────────┬────────────┘
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼

    Audio            Feature          Runtime

       │                 │                 │

   WavReader          FBank         ONNX Runtime

       │                 │                 │

       └─────────────────┼─────────────────┘
                         ▼

                  Acoustic Model

                         ▼

                     log_probs

                         ▼

                    CTC Decoder

                         ▼

                        Text
```

Recognizer 负责整个 Pipeline。

Runtime 只负责神经网络推理。

Decoder 只负责解码。

Feature 只负责特征。

各模块完全独立。

---

# 为什么这样设计

## 不重复造轮子

本项目**不会实现：**

- Conv1D
- BatchNorm
- GRU
- LSTM
- Transformer
- Tensor
- Layer Framework

原因：

这些属于：

> 深度学习框架开发

而不是：

> ASR Runtime 工程

以后工作中真正面对的是：

```
PyTorch

↓

Export ONNX

↓

TensorRT / ONNX Runtime

↓

C++ Runtime
```

因此项目重点放在：

- Runtime
- Pipeline
- Streaming
- 性能优化
- 工程架构

---

# 项目目录

```
offline_asr/

├── CMakeLists.txt
├── cmake/

├── include/offline_asr/

│
├── audio/
│   ├── wav_reader.h
│   ├── resampler.h
│   ├── ring_buffer.h
│   └── audio_stream.h
│
├── feature/
│   ├── fft.h
│   ├── stft.h
│   ├── mel_filterbank.h
│   └── fbank.h
│
├── runtime/
│   ├── runtime.h
│   ├── onnx_runtime.h
│   ├── runtime_factory.h
│   └── tensor.h
│
├── decoder/
│   ├── ctc_decoder.h
│   ├── beam_search.h
│   └── token_table.h
│
├── recognizer/
│   ├── recognizer.h
│   └── recognizer_config.h
│
├── vad/
│   ├── vad.h
│   └── silero_vad.h
│
├── benchmark/
│
├── profiler/
│
├── utils/
│   ├── logger.h
│   ├── timer.h
│   └── thread_pool.h
│
├── configs/
│
├── models/
│
├── examples/
│
├── tests/
│
└── docs/
```

---

# 模块说明

## Audio

负责：

- WAV读取
- PCM读取
- 重采样
- Streaming输入

接口：

```cpp
AudioReader

AudioStream

RingBuffer
```

---

## Feature

负责：

- FFT
- STFT
- Mel Filter
- FBank

目前已经完成：

- ✅ FFT
- ✅ Mel
- ✅ FBank

后续增加：

Benchmark。

例如：

```
FBank:

Average Latency

1.35 ms
```

---

## Runtime

新增重点模块。

抽象接口：

```cpp
class Runtime {
public:

    virtual bool LoadModel(...) = 0;

    virtual Tensor Forward(...) = 0;

    virtual ~Runtime() = default;
};
```

实现：

```
OnnxRuntime
```

以后：

```
TensorRTRuntime

OpenVINORuntime

NCNNRuntime
```

Recognizer 完全不用修改。

---

## Decoder

目前已有：

- CTC Decoder
- Prefix Beam Search

建议继续增加：

- Streaming Decoder
- Partial Result
- Timestamp（预留）

---

## Recognizer

整个系统唯一入口。

示例：

```cpp
Recognizer recognizer(config);

auto result = recognizer.Recognize("test.wav");
```

内部流程：

```
Read Audio

↓

FBank

↓

Runtime

↓

CTC Decoder

↓

Text
```

---

## Config

全部配置放入 YAML。

例如：

```yaml
model: models/conformer.onnx

sample_rate: 16000

threads: 8

beam_size: 10

vad: true
```

Recognizer：

```cpp
Recognizer recognizer("config.yaml");
```

无需重新编译。

---

## Logger

推荐：

spdlog。

统一：

```cpp
LOG_INFO()

LOG_WARN()

LOG_ERROR()
```

方便定位问题。

---

## Benchmark

增加：

```
FBank

Inference

Decoder

Recognizer
```

输出：

```
======================

FBank       1.42 ms

Inference  12.85 ms

Decoder     0.81 ms

Total      15.08 ms

RTF         0.12

======================
```

---

## Profiler

统计：

```
每个模块耗时

CPU占用

内存峰值

线程利用率
```

方便以后优化。

---

## CLI

支持：

```
offline_asr transcribe test.wav

offline_asr batch dataset/

offline_asr benchmark

offline_asr stream
```

---

## Streaming

Recognizer 新增：

```cpp
AcceptWaveform()

Decode()

GetResult()

Reset()
```

支持：

实时识别。

---

## VAD

建议加入：

Silero VAD。

Pipeline：

```
Audio

↓

VAD

↓

FBank

↓

Runtime

↓

Decoder
```

后续可以替换：

FSMN VAD。

---

# Python

Python 不负责 Runtime。

仅负责：

```
python/

export_onnx.py

infer.py

verify.py
```

作用：

- 导出 ONNX
- 数值验证
- 推理验证

**不实现训练代码。**

---

# Model

```
models/

conformer.onnx

tokens.txt

config.yaml
```

以后：

只替换：

```
conformer.onnx
```

即可支持新模型。

---

# Docs

建议维护：

```
Architecture.md

Runtime.md

Recognizer.md

Streaming.md

Benchmark.md

Performance.md
```

形成完整文档。

---

# Benchmark 指标

建议输出：

- RTF
- Latency
- Throughput
- Peak Memory
- CPU Usage
- GPU Usage（未来）

---

# 单元测试

GoogleTest：

```
Feature

Runtime

Decoder

Recognizer
```

新增：

Golden Test。

即：

同一个 wav：

```
结果不能变化。
```

CI 自动检查。

---

# CI

建议：

GitHub Actions：

- Build
- Unit Test
- Benchmark
- clang-format
- clang-tidy
- Sanitizer

---

# 后续扩展路线

## V1.0

完成：

```
Audio

↓

FBank

↓

ONNX Runtime

↓

CTC

↓

Text
```

---

## V1.1

增加：

```
Streaming

Benchmark

Profiler
```

---

## V1.2

增加：

```
Silero VAD

YAML Config

Logger
```

---

## V2.0

增加：

```
TensorRT Runtime

Batch Inference

Async Pipeline
```

---

## V3.0

支持：

```
Conformer

Paraformer

Whisper Encoder

Zipformer
```

无需修改 Recognizer。

---

# 推荐开发顺序

| 阶段 | 内容 | 输出 |
|------|------|------|
| 1 | 完善 Audio + FBank | Feature 模块 |
| 2 | 接入 ONNX Runtime | Runtime 模块 |
| 3 | Runtime + CTC 集成 | 完整识别 Pipeline |
| 4 | 实现 Recognizer API | SDK 初版 |
| 5 | YAML + Logger + CLI | 可发布版本 |
| 6 | Benchmark + Profiler | 性能评测 |
| 7 | Streaming 支持 | 流式识别 |
| 8 | VAD 集成 | 工业级 Pipeline |
| 9 | TensorRT Backend | 第二推理后端 |
| 10 | 多模型支持 | Runtime Framework |

---

# 最终目标

最终希望项目达到如下调用方式：

```cpp
Recognizer recognizer("config.yaml");

auto result = recognizer.Recognize("demo.wav");

std::cout << result.text << std::endl;
```

Recognizer 内部自动完成：

```
Audio

↓

FBank

↓

Runtime

↓

CTC Decoder

↓

Text
```

未来新增任何模型，仅需替换：

```
model.onnx
```

无需修改 Runtime 架构。

---

# 项目定位

> **OfflineASR**
>
> 一个面向工业部署的、可扩展的 C++17 Speech Runtime Framework。

它不是一个 Demo，也不是一个深度学习框架，而是一个能够持续扩展的 **语音推理 SDK**。未来可以逐步支持：

- ONNX Runtime
- TensorRT
- OpenVINO
- Streaming ASR
- VAD
- 多模型切换
- Batch 推理
- 性能分析

最终形成一个接近 Sherpa、FunASR Runtime、Kaldi Runtime 架构思想的个人工程项目。