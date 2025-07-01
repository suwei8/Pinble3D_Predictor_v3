import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tft_dataset import TFTDataset
from tft_model import LotteryTFT
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 解析参数 ===
mode = "full"
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        mode = sys.argv[idx + 1]

print(f"✅ 当前训练模式: {mode}")

# === 加载数据集 ===
dataset = TFTDataset(CSV_PATH, seq_len=10, mode=mode)
print(f"✅ 数据集可用样本数: {len(dataset)}")

if len(dataset) == 0:
    print("❌ 样本不足，跳过本次训练")
    sys.exit(0)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 初始化模型 ===
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

# === 加载最近权重 ===
latest_model = None
for f in sorted(os.listdir(MODEL_DIR), reverse=True):
    if f.startswith("tft_best") and f.endswith(".pth"):
        latest_model = os.path.join(MODEL_DIR, f)
        break

if latest_model:
    model.load_state_dict(torch.load(latest_model, map_location=device))
    print(f"✅ 已加载模型: {latest_model}")

# === 训练 ===
best_loss = float('inf')
max_epoch = 500 if mode == "full" else 3

for epoch in range(max_epoch):
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
        if mode == "incremental":
            save_name = os.path.join(MODEL_DIR, "tft_best_incremental.pth")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = os.path.join(MODEL_DIR, f"tft_best_{timestamp}.pth")
        torch.save(model.state_dict(), save_name)
        print(f"✅ 已保存: {save_name}")
