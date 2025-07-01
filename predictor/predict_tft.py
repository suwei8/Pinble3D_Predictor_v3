import torch
from tft_model import LotteryTFT
from tft_dataset import TFTDataset
import os
from glob import glob
import argparse

# === CLI 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="path to .pth checkpoint")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === 选模型 ===
if args.checkpoint:
    MODEL_PATH = os.path.join(MODEL_DIR, args.checkpoint)
else:
    pth_list = sorted(glob(os.path.join(MODEL_DIR, "tft_best_*.pth")))
    if pth_list:
        MODEL_PATH = pth_list[-1]
    else:
        MODEL_PATH = os.path.join(MODEL_DIR, "tft_best.pth")

print(f"✅ 已选模型: {MODEL_PATH}")

# === 加载数据 ===
dataset = TFTDataset(CSV_PATH, seq_len=10)
print(f"✅ 数据集大小: {len(dataset)}")

last_seq, y_reg, y_cls, y_seq = dataset[-1]
last_seq = last_seq.unsqueeze(0).to(device)
print(f"✅ last_seq shape: {last_seq.shape}")

model = LotteryTFT(
    input_dim=len(dataset.feature_cols),
    model_dim=32,
    nhead=4,
    num_layers=2
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    pred_reg, pred_cls, pred_seq = model(last_seq)

    pred_sum = pred_reg.item()
    pred_cls_digit = torch.argmax(pred_cls, dim=1).item()
    pred_seq_digits = [torch.argmax(pred_seq[:, i*10:(i+1)*10], dim=1).item() for i in range(3)]

print(f"🎯 预测和值: {pred_sum:.2f}")
print(f"🎯 预测独胆: {pred_cls_digit}")
print(f"🎯 预测试机号3位: {pred_seq_digits}")
