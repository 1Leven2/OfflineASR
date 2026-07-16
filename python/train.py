"""
CNN-BiGRU 声学模型训练脚本

支持:
  - AN4 真实数据集 (自动下载/手动放置)
  - 合成数字数据集 (SyntheticDigitsDataset, 无需下载)

用法:
  python train.py                          # 合成数据快速验证
  python train.py --dataset an4 --data_dir data/an4  # AN4 真实数据
  python train.py --dataset an4 --data_dir data/an4 --epochs 100 --batch_size 32
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import AcousticModel, count_parameters
from dataset import (
    AN4Dataset, LibriSpeechDataset, SyntheticDigitsDataset,
    collate_fn, VOCAB_SIZE, decode_ids, CHAR_TO_ID
)


def train_epoch(model, loader, optimizer, ctc_loss, device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        fbank = batch['fbank'].to(device)                 # [B, T, D]
        targets = batch['target'].to(device)              # [B, L]
        fbank_lengths = batch['fbank_lengths']            # [B]
        target_lengths = batch['target_lengths']          # [B]

        log_probs = model(fbank)                          # [B, T_out, V]
        # CTC Loss expects [T, B, V]
        log_probs_ctc = log_probs.transpose(0, 1)         # [T_out, B, V]

        output_lengths = model.output_lengths(fbank_lengths)

        loss = ctc_loss(
            log_probs_ctc,
            targets,
            output_lengths,
            target_lengths,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * fbank.size(0)
        total_items += fbank.size(0)

    return total_loss / total_items


@torch.no_grad()
def evaluate(model, loader, device):
    """评估: 计算 CER (Character Error Rate)"""
    model.eval()
    total_dist = 0
    total_len = 0

    for batch in loader:
        fbank = batch['fbank'].to(device)
        fbank_lengths = batch['fbank_lengths']
        texts = batch['text']

        log_probs = model(fbank)  # [B, T_out, V]
        output_lengths = model.output_lengths(fbank_lengths)

        for i in range(len(texts)):
            # Greedy decode
            logp = log_probs[i, :output_lengths[i]]  # [T', V]
            pred_ids = logp.argmax(dim=-1).tolist()
            pred_text = decode_ids(pred_ids)
            ref_text = texts[i]

            # Edit distance
            dist = edit_distance(pred_text, ref_text)
            total_dist += dist
            total_len += len(ref_text)

    cer = total_dist / total_len if total_len > 0 else float('inf')
    return cer


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def main():
    parser = argparse.ArgumentParser(description='Train CNN-BiGRU acoustic model')
    parser.add_argument('--dataset', default='synthetic',
                        choices=['synthetic', 'an4', 'librispeech'])
    parser.add_argument('--data_dir', default='data/an4')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    if args.dataset == 'synthetic':
        print("Using synthetic digit dataset (for quick testing)")
        train_ds = SyntheticDigitsDataset(num_samples=500, max_digits=3)
        test_ds = SyntheticDigitsDataset(num_samples=100, max_digits=3)
    elif args.dataset == 'an4':
        print(f"Loading AN4 from {args.data_dir}")
        train_ds = AN4Dataset(args.data_dir, split='train')
        test_ds = AN4Dataset(args.data_dir, split='test')
    elif args.dataset == 'librispeech':
        print(f"Loading LibriSpeech from {args.data_dir}")
        full_ds = LibriSpeechDataset(args.data_dir, split='test-clean')
        # 80/20 split since test-clean is the only subset we have
        n = len(full_ds)
        n_train = int(n * 0.8)
        train_ds = torch.utils.data.Subset(full_ds, range(n_train))
        test_ds = torch.utils.data.Subset(full_ds, range(n_train, n))
        print(f"  Total: {n}, Train: {n_train}, Test: {n - n_train}")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=args.num_workers)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Model
    model = AcousticModel(vocab_size=VOCAB_SIZE).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_cer = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            cer = evaluate(model, test_loader, device)
            status = '*' if cer < best_cer else ' '
            print(f"[{status}] Epoch {epoch:3d}/{args.epochs} | "
                  f"loss={train_loss:.4f} | CER={cer:.4f}")

            if cer < best_cer:
                best_cer = cer
                torch.save(model.state_dict(),
                           os.path.join(args.checkpoint_dir, 'best.pt'))
        else:
            print(f"    Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f}")

    print(f"\nBest CER: {best_cer:.4f}")
    print(f"Model saved to {args.checkpoint_dir}/best.pt")


if __name__ == '__main__':
    main()
