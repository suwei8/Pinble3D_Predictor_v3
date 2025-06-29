# predictor/train_real_incremental.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tft_dataset import TFTDataset
from tft_model import LotteryTFT
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 项目路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "tft_best.pth")

# === 加载数据 ===
dataset = TFTDataset(CSV_PATH, seq_len=10)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 初始化模型 ===
model = LotteryTFT(
    input_dim=len(dataset.feature_cols),
    model_dim=32,
    nhead=4,
    num_layers=2
).to(device)

# === 若存在历史模型则加载 ===
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 已加载历史模型: {MODEL_PATH}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
loss_seq = nn.CrossEntropyLoss()

best_loss = float('inf')

# === 如果有历史 loss 可加载 ===
# 这里简单写死，若有更完善的记录可以保存到 checkpoint 中
if os.path.exists(MODEL_PATH):
    best_loss = 1e8  # 也可以从外部文件记录恢复

# === 训练循环 ===
for epoch in range(50):  # 继续跑 50 轮，可自行改
    model.train()
    total_loss = 0

    for seq_x, y_reg, y_cls, y_seq in train_loader:
        seq_x, y_reg, y_cls, y_seq = seq_x.to(device), y_reg.to(device), y_cls.to(device), y_seq.to(device)

        optimizer.zero_grad()
        pred_reg, pred_cls, pred_seq = model(seq_x)

        loss1 = loss_mse(pred_reg.squeeze(), y_reg)
        loss2 = loss_ce(pred_cls, y_cls)
        loss3 = sum([loss_seq(pred_seq[:, i*10:(i+1)*10], y_seq[:, i]) for i in range(3)])
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    # 确保保存目录
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ 已保存增量最优模型: {MODEL_PATH}")
