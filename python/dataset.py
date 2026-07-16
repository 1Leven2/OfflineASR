"""
AN4 + LibriSpeech 数据集加载

AN4: CMU 数字串数据集，适合快速验证
LibriSpeech: 100h 干净朗读语音，训练通用模型

统一输出: FBank 特征 [T, 80] + token 序列
"""
import os
import re
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Optional

# 字符级词汇表: blank(0) + space(1) + '(2) + a-z(3-28)
BLANK = '<blank>'
SPACE = '<space>'
CHAR_VOCAB = [BLANK, SPACE, "'"] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = CHAR_VOCAB
VOCAB_SIZE = len(CHAR_VOCAB)  # 29


class AudioTransform:
    """音频 → FBank 特征 [T, num_mel_bins]"""
    def __init__(self, sample_rate: int = 16000, num_mel_bins: int = 80):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins

    def __call__(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(0, keepdim=True)  # 多声道 → 单声道
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # 归一化
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=25.0,
            frame_shift=10.0,
            sample_frequency=self.sample_rate,
            window_type='povey',
            preemphasis_coefficient=0.97,
            use_energy=False,
        )
        return fbank  # [T, num_mel_bins]


def encode_text(text: str) -> list[int]:
    """文本 → token ID 列表"""
    tokens = []
    text = text.lower()
    for ch in text:
        if ch in CHAR_TO_ID:
            tokens.append(CHAR_TO_ID[ch])
    return tokens


def decode_ids(ids: list[int]) -> str:
    """token ID 列表 → 文本 (CTC collapse)"""
    result = []
    prev = -1
    for idx in ids:
        if idx == 0:  # blank
            prev = -1
            continue
        if idx == prev:  # merge duplicates
            continue
        result.append(ID_TO_CHAR[idx])
        prev = idx
    return ''.join(result)


class AN4Dataset(torch.utils.data.Dataset):
    """
    AN4 数字串数据集

    期望目录结构:
        data/an4/
          train/
            *.wav (或 train.txt 包含 wav→transcript 映射)
          test/
            *.wav
    """
    labels = {
        'ZERO': 'zero', 'ONE': 'one', 'TWO': 'two', 'THREE': 'three',
        'FOUR': 'four', 'FIVE': 'five', 'SIX': 'six', 'SEVEN': 'seven',
        'EIGHT': 'eight', 'NINE': 'nine', 'OH': 'oh',
    }

    def __init__(self, root: str, split: str = 'train', download: bool = False):
        self.root = Path(root)
        self.split = split
        self.transform = AudioTransform()

        # 查找 WAV 文件
        self.samples: list[tuple[Path, str]] = []  # (path, transcript)
        wav_dir = self.root / split

        if wav_dir.exists():
            wav_files = sorted(wav_dir.glob('*.wav'))
            for wav_path in wav_files:
                transcript = self._infer_transcript(wav_path.stem)
                if transcript:
                    self.samples.append((wav_path, transcript))
        else:
            # 尝试扫描整个目录
            for wav_path in sorted(Path(root).rglob('*.wav')):
                transcript = self._infer_transcript(wav_path.stem)
                if transcript:
                    self.samples.append((wav_path, transcript))

        if not self.samples:
            raise RuntimeError(
                f"No WAV files found in {root}/{split}. "
                f"Please download AN4: https://catalog.ldc.upenn.edu/LDC93S1"
            )

    def _infer_transcript(self, stem: str) -> Optional[str]:
        """从文件名推断转录文本 (AN4 命名为 AN4_digits_xxx)"""
        # 尝试解析 AN4 命名格式: "xxx_1234"
        parts = re.split(r'[_\s]+', stem)
        words = []
        for p in parts:
            upper = p.upper()
            if upper in self.labels:
                words.append(self.labels[upper])
            elif p.isdigit():
                for d in p:
                    digit_map = {str(i): self.labels[k] for i, k in
                                 enumerate(['ZERO','ONE','TWO','THREE','FOUR',
                                           'FIVE','SIX','SEVEN','EIGHT','NINE'])}
                    words.append(digit_map.get(d, d))
        return ' '.join(words) if words else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        waveform, sr = sf.read(str(path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

        fbank = self.transform(waveform, sr)
        tokens = encode_text(text)

        return fbank, torch.tensor(tokens, dtype=torch.long), text


class LibriSpeechDataset(torch.utils.data.Dataset):
    """
    LibriSpeech 数据集 (需先用 torchaudio 下载)

    目录结构: data/LibriSpeech/<subset>/<speaker>/<chapter>/*.flac + *.trans.txt
    子集: dev-clean, test-clean, train-clean-100, train-clean-360 等
    """

    def __init__(self, root: str, split: str = 'test-clean', cache_fbank: bool = True):
        self.root = Path(root) / 'LibriSpeech' / split
        self.transform = AudioTransform()
        self.samples: list[tuple[Path, str]] = []
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, str]] = {}
        self._use_cache = cache_fbank

        if not self.root.exists():
            raise RuntimeError(
                f"LibriSpeech {split} not found at {self.root}. "
                f"Download: torchaudio.datasets.LIBRISPEECH('{root}', "
                f"url='{split}', download=True)"
            )

        # LibriSpeech: <root>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
        for trans_file in sorted(self.root.rglob('*.trans.txt')):
            chapter_dir = trans_file.parent
            with open(trans_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    # FLAC file: <chapter_dir>/<utt_id>.flac
                    flac_path = chapter_dir / f'{utt_id}.flac'
                    if flac_path.exists():
                        self.samples.append((flac_path, text.lower()))

        if not self.samples:
            raise RuntimeError(f"No FLAC+transcript pairs found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        path, text = self.samples[idx]
        waveform, sr = sf.read(str(path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

        fbank = self.transform(waveform, sr)
        tokens = encode_text(text)

        result = (fbank, torch.tensor(tokens, dtype=torch.long), text)
        if self._use_cache:
            self._cache[idx] = result
        return result


class SyntheticDigitsDataset(torch.utils.data.Dataset):
    """合成数字音频数据集 (用于测试，无需下载任何数据)"""

    DIGITS = {
        'zero': 'zero', 'one': 'one', 'two': 'two', 'three': 'three',
        'four': 'four', 'five': 'five', 'six': 'six', 'seven': 'seven',
        'eight': 'eight', 'nine': 'nine',
    }

    def __init__(self, num_samples: int = 200, max_digits: int = 3,
                 sample_rate: int = 16000):
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.sample_rate = sample_rate
        self.transform = AudioTransform(sample_rate)

        self.digit_texts = list(self.DIGITS.values())
        self._generate()

    def _generate(self):
        import random
        random.seed(42)
        self.data: list[tuple[torch.Tensor, str]] = []
        for _ in range(self.num_samples):
            n_digits = random.randint(1, self.max_digits)
            digits = [random.choice(self.digit_texts) for _ in range(n_digits)]
            text = ' '.join(digits)
            # 合成简单正弦波 (不同数字用不同频率)
            duration = random.uniform(0.4, 0.8) * n_digits
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.zeros_like(t)
            for i, d in enumerate(digits):
                start = int(i * self.sample_rate * 0.5)
                end = start + int(self.sample_rate * 0.45)
                if end > len(t):
                    end = len(t)
                freq_map = {d: 200 + 100 * i for i, d in enumerate(self.digit_texts)}
                freq = freq_map.get(d, 440)
                audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * t[start:end])

            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            self.data.append((waveform, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, text = self.data[idx]
        fbank = self.transform(waveform, self.sample_rate)
        tokens = encode_text(text)
        return fbank, torch.tensor(tokens, dtype=torch.long), text


def collate_fn(batch):
    """变长批处理: padding + 长度记录"""
    fbanks, targets, texts = zip(*batch)

    fbank_lengths = torch.tensor([f.shape[0] for f in fbanks], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    max_frames = fbank_lengths.max().item()
    max_tokens = target_lengths.max().item()

    feat_dim = fbanks[0].shape[1]
    fbank_padded = torch.zeros(len(batch), max_frames, feat_dim)
    target_padded = torch.zeros(len(batch), max_tokens, dtype=torch.long)

    for i, (f, t) in enumerate(zip(fbanks, targets)):
        fbank_padded[i, :f.shape[0]] = f
        target_padded[i, :t.shape[0]] = t

    return {
        'fbank': fbank_padded,
        'fbank_lengths': fbank_lengths,
        'target': target_padded,
        'target_lengths': target_lengths,
        'text': texts,
    }


if __name__ == '__main__':
    # 测试合成数据集
    ds = SyntheticDigitsDataset(num_samples=5, max_digits=2)
    for i in range(3):
        fbank, tokens, text = ds[i]
        print(f"[{i}] text='{text}' → tokens={tokens.tolist()}, fbank shape={fbank.shape}")

    # 测试 collate
    loader = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch: fbank={batch['fbank'].shape}, target={batch['target'].shape}")
    print(f"Lengths: fbank={batch['fbank_lengths']}, target={batch['target_lengths']}")
