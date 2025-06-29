# predictor/tft_dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class TFTDataset(Dataset):
    def __init__(self, csv_path, seq_len=10):
        self.df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
        self.seq_len = seq_len

        # ✅ 自动找特征列
        self.feature_cols = [col for col in self.df.columns if col not in ['date', 'issue', 'single_digit', 'sim_digit_1', 'sim_digit_2', 'sim_digit_3']]

        print(f"✅ 数据集大小: {len(self.df)} 条")
        print(f"✅ 特征列: {self.feature_cols}")

    def __len__(self):
        # ✅ 防止取到不足序列
        return max(0, len(self.df) - self.seq_len)

    def __getitem__(self, idx):
        # ✅ 兜底处理
        if idx < 0:
            idx = 0
        if idx > len(self.df) - self.seq_len - 1:
            idx = len(self.df) - self.seq_len - 1

        seq_x = torch.tensor(self.df.loc[idx:idx+self.seq_len-1, self.feature_cols].values, dtype=torch.float)
        y_reg = torch.tensor(self.df.loc[idx+self.seq_len, 'sim_sum_val'], dtype=torch.float)
        y_cls = torch.tensor(self.df.loc[idx+self.seq_len, 'single_digit'], dtype=torch.long)
        y_seq = torch.tensor([
            self.df.loc[idx+self.seq_len, 'sim_digit_1'],
            self.df.loc[idx+self.seq_len, 'sim_digit_2'],
            self.df.loc[idx+self.seq_len, 'sim_digit_3']
        ], dtype=torch.long)

        return seq_x, y_reg, y_cls, y_seq
