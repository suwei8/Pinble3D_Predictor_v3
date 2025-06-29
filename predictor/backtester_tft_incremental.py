# predictor/backtester_tft_incremental.py

import torch
from tft_model import LotteryTFT
from tft_dataset import TFTDataset
import os
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === 自动寻找最新模型 ===
pth_list = sorted(glob(os.path.join(MODEL_DIR, "tft_best_*.pth")))
if pth_list:
    MODEL_PATH = pth_list[-1]
else:
    MODEL_PATH = os.path.join(MODEL_DIR, "tft_best.pth")
print(f"✅ 已选模型: {MODEL_PATH}")

# === 加载数据 ===
dataset = TFTDataset(CSV_PATH, seq_len=10)
print(f"✅ 数据集大小: {len(dataset)}")

model = LotteryTFT(
    input_dim=len(dataset.feature_cols),
    model_dim=32,
    nhead=4,
    num_layers=2
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === 回测最近 N ===
N = 20
hit_cls = 0
seq_hit_total = 0

print(f"===== 🎯 TFT 回测最近 {N} 期 =====")

with torch.no_grad():
    for idx in range(len(dataset) - N, len(dataset)):
        seq_x, y_reg, y_cls, y_seq = dataset[idx]
        seq_x = seq_x.unsqueeze(0).to(device)

        pred_reg, pred_cls, pred_seq = model(seq_x)

        pred_cls_digit = torch.argmax(pred_cls, dim=1).item()
        pred_seq_digits = [torch.argmax(pred_seq[:, i*10:(i+1)*10], dim=1).item() for i in range(3)]

        cls_hit = pred_cls_digit == y_cls.item()
        seq_hit_count = sum([int(d == y_seq[i].item()) for i, d in enumerate(pred_seq_digits)])

        if cls_hit:
            hit_cls += 1
        seq_hit_total += seq_hit_count

        print(f"期:{idx} | 独胆:{pred_cls_digit} 真:{y_cls.item()} | 命中:{'✔️' if cls_hit else '❌'} | "
              f"试机号:{pred_seq_digits} 真:{[y_seq[i].item() for i in range(3)]} | 命中位:{seq_hit_count}")

print("\n===== ✅ 回测统计 =====")
print(f"独胆命中率: {hit_cls}/{N} = {hit_cls/N:.2%}")
print(f"试机号平均命中位数: {seq_hit_total}/{N} = {seq_hit_total/N:.2f}")
