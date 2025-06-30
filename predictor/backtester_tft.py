import matplotlib.pyplot as plt
import torch
from tft_model import LotteryTFT
from tft_dataset import TFTDataset
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "tft_best.pth")

# === 加载数据 ===
dataset = TFTDataset(CSV_PATH, seq_len=10)

# === 加载模型 ===
model = LotteryTFT(
    input_dim=len(dataset.feature_cols),
    model_dim=32,
    nhead=4,
    num_layers=2
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === 回测 ===
correct_hits = 0
total_samples = 200
hit_rate_per_epoch = []

for idx in range(len(dataset) - total_samples, len(dataset)):
    seq_x, y_reg, y_cls, y_seq = dataset[idx]
    seq_x = seq_x.unsqueeze(0).to(device)  # 增加batch维度

    with torch.no_grad():
        pred_reg, pred_cls, pred_seq = model(seq_x)

        # 预测
        pred_cls_digit = torch.argmax(pred_cls, dim=1).item()
        true_digit = y_cls.item()

        if pred_cls_digit == true_digit:
            correct_hits += 1

        # 记录
        hit_rate_per_epoch.append(correct_hits / (idx + 1))

# === 绘图 ===
plt.plot(range(total_samples), hit_rate_per_epoch, label="Hit Rate")
plt.xlabel('Samples')
plt.ylabel('Hit Rate')
plt.title('Prediction Hit Rate over Time')
plt.legend()
plt.show()
