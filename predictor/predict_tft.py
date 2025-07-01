import pandas as pd
import torch
from torch.utils.data import Dataset

class TFTDataset(Dataset):
    def __init__(self, csv_path, seq_len=10, mode="full"):
        self.mode = mode
        self.seq_len = seq_len

        df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

        # === Label Encoding ===
        df['sim_pattern'] = df['sim_pattern'].map({'组六': 0, '组三': 1, '豹子': 2})
        df['open_pattern'] = df['open_pattern'].map({'组六': 0, '组三': 1, '豹子': 2})

        if self.mode == "incremental":
            if len(df) < seq_len:
                print(f"⚠️ 增量模式: 样本数 {len(df)} < seq_len={seq_len}，自动使用全量")
                self.df = df
            else:
                self.df = df.tail(seq_len).reset_index(drop=True)
                print(f"✅ 增量模式: 使用最近 {seq_len} 条")
        else:
            self.df = df
            print(f"✅ 全量模式: 样本数 {len(df)}")

        self.seq_len = min(len(self.df), seq_len)

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
        l = len(self.df) - self.seq_len
        return max(0, l)

    def __getitem__(self, idx):
        if idx + self.seq_len > len(self.df):
            raise IndexError(f"❌ idx={idx} 太小，无法构成序列")

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