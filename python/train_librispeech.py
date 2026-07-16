#!/usr/bin/env python3
"""LibriSpeech training — run from python/ directory"""
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, time

from dataset import LibriSpeechDataset, collate_fn, decode_ids, VOCAB_SIZE
from model import AcousticModel, count_parameters
from train import train_epoch, evaluate

# Log to file to avoid stdout buffering issues
log = open('training.log', 'w', buffering=1)  # line-buffered

def log_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    log.write(msg + '\n')
    log.flush()
    print(msg, flush=True)

# ---- Fast collate with truncation ----
def collate_truncated(batch, max_frames=1500):
    """collate_fn with max frame limit to avoid padding to extreme lengths"""
    fbanks, targets, texts = zip(*batch)

    fbank_lengths = torch.tensor([min(f.shape[0], max_frames) for f in fbanks], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    actual_max = fbank_lengths.max().item()
    max_tokens = target_lengths.max().item()
    feat_dim = fbanks[0].shape[1]

    fbank_padded = torch.zeros(len(batch), actual_max, feat_dim)
    target_padded = torch.zeros(len(batch), max_tokens, dtype=torch.long)

    for i, (f, t) in enumerate(zip(fbanks, targets)):
        fl = min(f.shape[0], max_frames)
        fbank_padded[i, :fl] = f[:fl]
        target_padded[i, :t.shape[0]] = t

    return {
        'fbank': fbank_padded,
        'fbank_lengths': fbank_lengths,
        'target': target_padded,
        'target_lengths': target_lengths,
        'text': texts,
    }

device = torch.device('cpu')
log_print(f"Device: {device}")

# Dataset
ds = LibriSpeechDataset('data', split='test-clean')
n = len(ds)
n_train = int(n * 0.8)
train_ds = torch.utils.data.Subset(ds, range(n_train))
test_ds = torch.utils.data.Subset(ds, range(n_train, n))
log_print(f"Total: {n}, Train: {n_train}, Test: {n - n_train}")

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=8, shuffle=True, collate_fn=collate_truncated,
    num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=8, shuffle=False, collate_fn=collate_truncated,
    num_workers=0)

# Model
model = AcousticModel(vocab_size=VOCAB_SIZE).to(device)
log_print(f"Parameters: {count_parameters(model):,}")

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

os.makedirs('checkpoints', exist_ok=True)
best_cer = float('inf')

# Pre-cache FBank
log_print("Pre-caching FBank features...")
t0 = time.time()
for i in range(len(ds)):
    _ = ds[i]
    if (i + 1) % 500 == 0:
        log_print(f"  Cached {i+1}/{n}")
log_print(f"Cache done in {time.time()-t0:.1f}s")

# Train
for epoch in range(1, 21):
    t_epoch = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        cer = evaluate(model, test_loader, device)
        status = '*' if cer < best_cer else ' '
        log_print(f"[{status}] Epoch {epoch:3d}/20 | loss={train_loss:.4f} | CER={cer:.4f} | {time.time()-t_epoch:.0f}s")
        if cer < best_cer:
            best_cer = cer
            torch.save(model.state_dict(), 'checkpoints/best.pt')
            log_print(f"  -> saved best.pt (CER={best_cer:.4f})")
    else:
        log_print(f"    Epoch {epoch:3d}/20 | loss={train_loss:.4f} | {time.time()-t_epoch:.0f}s")

log_print(f"\nBest CER: {best_cer:.4f}")
log.close()
