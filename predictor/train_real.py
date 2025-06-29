import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tft_dataset import TFTDataset
from tft_model import LotteryTFT
import os
from datetime import datetime

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 加载数据 ===
dataset = TFTDataset(CSV_PATH, seq_len=10)
print(f"✅ 数据集大小: {len(dataset)}")
print(f"✅ 特征列: {dataset.feature_cols}")

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 实例化模型 ===
model = LotteryTFT(
    input_dim=len(dataset.feature_cols),
    model_dim=32,
    nhead=4,
    num_layers=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
loss_seq = nn.CrossEntropyLoss()

# === 自动 Resume ===
latest_model = None
for f in sorted(os.listdir(MODEL_DIR), reverse=True):
    if f.startswith("tft_best") and f.endswith(".pth"):
        latest_model = os.path.join(MODEL_DIR, f)
        break

if latest_model:
    model.load_state_dict(torch.load(latest_model, map_location=device))
    print(f"✅ 已加载模型: {latest_model}")
else:
    print("❌ 未找到模型，初始化新模型")

# === 训练 ===
best_loss = float('inf')

for epoch in range(1000):  # 可改 epoch 数
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

    if avg_loss < best_loss:
        best_loss = avg_loss
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(MODEL_DIR, f"tft_best_{timestamp}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"已保存 {best_model_path}")
