import pandas as pd
import torch
from torch.utils.data import Dataset

class TFTDataset(Dataset):
    def __init__(self, csv_path, seq_len=10, mode="full"):
        self.seq_len = seq_len
        self.mode = mode

        self.df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        # === ✅ Label Encoding ===
        self.df['sim_pattern'] = self.df['sim_pattern'].map({'组六': 0, '组三': 1, '豹子': 2})
        self.df['open_pattern'] = self.df['open_pattern'].map({'组六': 0, '组三': 1, '豹子': 2})

        # === ✅ 如果是增量模式，只取最新 seq_len 条 ===
        if self.mode == "incremental":
            self.df = self.df.tail(self.seq_len).reset_index(drop=True)
            print(f"✅ 增量模式启用，仅使用最近 {self.seq_len} 条")
        else:
            print(f"✅ 全量模式，样本数: {len(self.df)}")

        # === ✅ 用到的特征列 ===
        self.feature_cols = [
            'sim_test_code', 'open_code',
            'sim_sum_val', 'sim_span',
            'open_digit_1', 'open_digit_2', 'open_digit_3',
            'open_sum_val', 'open_span',
            'sim_pattern', 'open_pattern',
            'sim_pattern_组三', 'open_pattern_组三',
            'sim_pattern_组六', 'open_pattern_组六',
            'sim_pattern_豹子', 'open_pattern_豹子',
            'match_count', 'match_pos_count',
            'single_hot_5', 'single_hot_3'
        ]

        print(f"✅ 特征列: {self.feature_cols}")

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        if idx + self.seq_len > len(self.df):
            raise IndexError(f"❌ idx={idx} 太小，无法构成有效序列")

        seq_x = torch.tensor(
            self.df.loc[idx:idx+self.seq_len-1, self.feature_cols].values,
            dtype=torch.float
        )

        y_reg = torch.tensor(self.df.loc[idx+self.seq_len-1, 'open_sum_val'], dtype=torch.float)
        y_cls = torch.tensor(self.df.loc[idx+self.seq_len-1, 'single_digit'], dtype=torch.long)
        y_seq = torch.tensor([
            self.df.loc[idx+self.seq_len-1, 'sim_digit_1'],
            self.df.loc[idx+self.seq_len-1, 'sim_digit_2'],
            self.df.loc[idx+self.seq_len-1, 'sim_digit_3']
        ], dtype=torch.long)

        return seq_x, y_reg, y_cls, y_seq
