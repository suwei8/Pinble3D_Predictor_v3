# predictor/train_real.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tft_dataset import TFTDataset
from tft_model import LotteryTFT

import os
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 加载数据 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

dataset = TFTDataset(CSV_PATH, seq_len=10)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 实例化模型 ===
MODEL_PATH = os.path.join(MODEL_DIR, "tft_best.pth")

model = LotteryTFT(
    input_dim=len(dataset.feature_cols),
    model_dim=32,
    nhead=4,
    num_layers=2
).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 已加载模型: {MODEL_PATH}")
else:
    print("❌ 未找到模型，初始化新模型")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # 每10个epoch衰减

loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
loss_seq = nn.CrossEntropyLoss()

best_loss = float('inf')
log_path = os.path.join(MODEL_DIR, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

with open(log_path, "w") as log_file:
    for epoch in range(200):  # 🔥 可长时间运行
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
        scheduler.step()

        log_str = f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        print(log_str)
        log_file.write(log_str + "\n")

        # 保存最优
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_name = f"tft_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            best_model_path = os.path.join(MODEL_DIR, best_model_name)
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), MODEL_PATH)  # 同步最新
            log_file.write(f"✅ 已保存 {best_model_path}\n")
            print(f"✅ 已保存 {best_model_path}")

print(f"🎉 完成训练！最优 Loss: {best_loss:.4f}")
