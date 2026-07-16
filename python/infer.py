"""
ONNX Runtime Python 推理验证

用法:
  python infer.py --model ../models/model.onnx --audio test.wav
"""
import argparse
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio


def extract_fbank(audio_path: str, sample_rate: int = 16000,
                  num_mel_bins: int = 80) -> np.ndarray:
    """提取 FBank 特征"""
    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=sample_rate,
        window_type='povey',
        preemphasis_coefficient=0.97,
        use_energy=False,
    )
    return fbank.numpy()  # [T, 80]


def decode_ctc(log_probs: np.ndarray, blank_id: int = 0) -> tuple[list[int], str]:
    """CTC 贪心解码"""
    tokens = log_probs.argmax(axis=-1)  # [T]

    # CTC collapse
    result = []
    prev = -1
    for t in tokens:
        if t == blank_id:
            prev = -1
            continue
        if t == prev:
            continue
        result.append(int(t))
        prev = t
    return result, ids_to_text(result)


def ids_to_text(ids: list[int]) -> str:
    """token ID → 文本"""
    chars = ['<blank>', '<space>', "'"] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    return ''.join(chars[i] for i in ids).replace('<space>', ' ')


def main():
    parser = argparse.ArgumentParser(description='ONNX Runtime inference test')
    parser.add_argument('--model', default='../models/model.onnx')
    parser.add_argument('--audio', help='Audio file path')
    parser.add_argument('--num_mel_bins', type=int, default=80)
    args = parser.parse_args()

    # Load ONNX model
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model: {args.model}")
    print(f"  Input: {input_name} {session.get_inputs()[0].shape}")
    print(f"  Output: {output_name}")

    if args.audio:
        # Extract features and run inference
        fbank = extract_fbank(args.audio, num_mel_bins=args.num_mel_bins)
        print(f"  FBank: {fbank.shape}")

        fbank_input = fbank[np.newaxis, :, :]  # [1, T, 80]
        log_probs = session.run([output_name], {input_name: fbank_input.astype(np.float32)})[0]
        log_probs = log_probs[0]  # [T_out, V]
        print(f"  Log probs: {log_probs.shape}")

        tokens, text = decode_ctc(log_probs)
        print(f"  Tokens: {tokens}")
        print(f"  Text: '{text}'")
    else:
        # Benchmark with random input
        print("No audio provided, running benchmark...")
        import time
        fbank = np.random.randn(1, 100, 80).astype(np.float32)

        # Warm-up
        for _ in range(10):
            _ = session.run([output_name], {input_name: fbank})

        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.time()
            _ = session.run([output_name], {input_name: fbank})
            times.append(time.time() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"  Avg latency: {avg_ms:.2f} ms")


if __name__ == '__main__':
    main()
