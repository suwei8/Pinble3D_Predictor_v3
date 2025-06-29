# predictor/tft_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ========= 1️⃣ Positional Encoding ===========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# ========= 2️⃣ 简易 Transformer Model ==========

class LotteryTFT(nn.Module):
    def __init__(self, input_dim, model_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # 多头输出
        self.head_reg = nn.Linear(model_dim, 1)   # 回归，如和值
        self.head_cls = nn.Linear(model_dim, 10)  # 分类，如独胆 (0-9)
        self.head_seq = nn.Linear(model_dim, 3 * 10)  # 三位号，multi-class

    def forward(self, x):
        x = self.input_proj(x)           # (B, T, D)
        x = self.pos_enc(x)              # (B, T, D)
        x = self.transformer(x)          # (B, T, D)

        pooled = x.mean(dim=1)           # 简单平均池化

        out_reg = self.head_reg(pooled).squeeze(-1)  # (B,)
        out_cls = self.head_cls(pooled)              # (B, 10)
        out_seq = self.head_seq(pooled)              # (B, 30)

        return out_reg, out_cls, out_seq

# ========= 3️⃣ 伪造样本 ===========
def generate_dummy_batch(batch_size=8, seq_len=20, input_dim=12):
    X = torch.rand(batch_size, seq_len, input_dim)
    y_reg = torch.rand(batch_size) * 27  # 和值可在 0-27
    y_cls = torch.randint(0, 10, (batch_size,))
    y_seq = torch.randint(0, 10, (batch_size, 3))
    return X, y_reg, y_cls, y_seq

# ========= 4️⃣ 训练循环示例 ===========
def train_dummy():
    model = LotteryTFT(input_dim=12)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seq = nn.CrossEntropyLoss()

    for step in range(50):
        X, y_reg, y_cls, y_seq = generate_dummy_batch()

        out_reg, out_cls, out_seq = model(X)

        loss_reg = criterion_reg(out_reg, y_reg)
        loss_cls = criterion_cls(out_cls, y_cls)
        loss_seq = sum([criterion_seq(out_seq[:, i*10:(i+1)*10], y_seq[:, i]) for i in range(3)])

        loss = loss_reg + loss_cls + loss_seq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Reg: {loss_reg.item():.4f} | Cls: {loss_cls.item():.4f} | Seq: {loss_seq.item():.4f}")

if __name__ == "__main__":
    train_dummy()
