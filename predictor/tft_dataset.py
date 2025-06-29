# predictor/tft_dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter

class TFTDataset(Dataset):
    def __init__(self, csv_path, seq_len=10):
        self.df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        # 加滚动热度特征
        self.df['single_hot_5'] = self.df['single_digit'].rolling(5, min_periods=1).apply(lambda x: Counter(x).most_common(1)[0][1])
        self.df['single_hot_3'] = self.df['single_digit'].rolling(3, min_periods=1).apply(lambda x: Counter(x).most_common(1)[0][1])

        # 填充可能缺的 pattern
        for col in [
            'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子',
            'open_pattern_组三', 'open_pattern_组六', 'open_pattern_豹子'
        ]:
            if col not in self.df.columns:
                self.df[col] = 0

        self.seq_len = seq_len

        self.feature_cols = [
            'sim_sum_val', 'sim_span',
            'open_digit_1', 'open_digit_2', 'open_digit_3',
            'open_sum_val', 'open_span',
            'match_count', 'match_pos_count',
            'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子',
            'open_pattern_组三', 'open_pattern_组六', 'open_pattern_豹子',
            'single_hot_5', 'single_hot_3'
        ]

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        if idx < self.seq_len - 1:
            raise IndexError(f"❌ idx={idx} 太小，无法构成有效序列")
        start = idx - self.seq_len + 1
        seq_x = self.df.iloc[start: idx + 1][self.feature_cols].values

        if len(seq_x) < self.seq_len:
            raise ValueError(f"❌ 实际切片长度 {len(seq_x)} < 预期 {self.seq_len}")

        seq_x = torch.tensor(seq_x, dtype=torch.float32)

        # 以下是你的标签
        y_reg = torch.tensor(self.df.iloc[idx]['sim_sum_val'], dtype=torch.float32)
        y_cls = torch.tensor(self.df.iloc[idx]['single_digit'], dtype=torch.long)
        y_seq = torch.tensor([int(x) for x in str(int(self.df.iloc[idx]['sim_test_code'])).zfill(3)], dtype=torch.long)

        return seq_x, y_reg, y_cls, y_seq
