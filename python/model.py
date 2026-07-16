"""
CNN-BiGRU 声学模型 (PyTorch)

架构:
  FBank [B, T, 80]
    → Conv1d(80→256, k=3) → BN → ReLU
    → Conv1d(256→256, k=3) → BN → ReLU → MaxPool1d(2)
    → Conv1d(256→384, k=3) → BN → ReLU
    → Conv1d(384→384, k=3) → BN → ReLU → MaxPool1d(2)
    → BiGRU(384→256, 2 layers)  → [B, T/4, 512]
    → Linear(512→29)            → [B, T/4, 29]
    → LogSoftmax(dim=-1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticModel(nn.Module):
    def __init__(self,
                 input_dim: int = 80,
                 vocab_size: int = 29,
                 cnn_dim1: int = 256,
                 cnn_dim2: int = 384,
                 rnn_hidden: int = 256,
                 rnn_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size

        # CNN frontend
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim1, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_dim1, cnn_dim1, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(cnn_dim1, cnn_dim2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim2),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_dim2, cnn_dim2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # RNN
        self.rnn = nn.GRU(
            input_size=cnn_dim2,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Output projection
        self.fc = nn.Linear(rnn_hidden * 2, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] FBank features (batch_first)
        Returns:
            [B, T_out, vocab_size] log probabilities
        """
        x = x.transpose(1, 2)  # [B, D, T] for Conv1d
        x = self.cnn(x)        # [B, C2, T/4]
        x = x.transpose(1, 2)  # [B, T/4, C2] for RNN

        x, _ = self.rnn(x)     # [B, T/4, 2*H]
        x = self.fc(x)         # [B, T/4, V]
        x = F.log_softmax(x, dim=-1)

        return x

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """输入帧数 → 输出帧数 (除以4，因为两次 MaxPool1d(2))"""
        return torch.div(input_lengths, 4, rounding_mode='floor')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = AcousticModel()
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward
    x = torch.randn(2, 100, 80)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    print(f"Output sum=1 check: {(y.exp().sum(-1) - 1.0).abs().max():.6f}")
