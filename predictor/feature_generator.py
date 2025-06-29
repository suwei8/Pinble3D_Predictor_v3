# predictor/feature_generator.py

import pandas as pd
import numpy as np
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIS_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")

print(f"✅ 加载原始: {HIS_PATH}")
df = pd.read_csv(HIS_PATH).dropna().reset_index(drop=True)

# === 基础字段 ===
df['sim_sum_val'] = df['sim_test_code'].astype(str).str.zfill(3).apply(lambda x: sum([int(c) for c in x]))
df['sim_span'] = df['sim_test_code'].astype(str).str.zfill(3).apply(lambda x: abs(int(max(x)) - int(min(x))))

# === 开奖拆分 ===
df['open_digit_1'] = df['open_code'].astype(str).str.zfill(3).str[0].astype(int)
df['open_digit_2'] = df['open_code'].astype(str).str.zfill(3).str[1].astype(int)
df['open_digit_3'] = df['open_code'].astype(str).str.zfill(3).str[2].astype(int)

df['open_sum_val'] = df['open_code'].astype(str).str.zfill(3).apply(lambda x: sum([int(c) for c in x]))
df['open_span'] = df['open_code'].astype(str).str.zfill(3).apply(lambda x: abs(int(max(x)) - int(min(x))))

# === 是否组三组六豹子 ===
def pattern(x):
    digits = list(x)
    if len(set(digits)) == 1:
        return '豹子'
    elif len(set(digits)) == 2:
        return '组三'
    else:
        return '组六'

df['sim_pattern'] = df['sim_test_code'].astype(str).str.zfill(3).apply(pattern)
df['open_pattern'] = df['open_code'].astype(str).str.zfill(3).apply(pattern)

for p in ['组三', '组六', '豹子']:
    df[f'sim_pattern_{p}'] = (df['sim_pattern'] == p).astype(int)
    df[f'open_pattern_{p}'] = (df['open_pattern'] == p).astype(int)

# === 交集匹配 ===
def match_count(row):
    s = set(str(row['sim_test_code']).zfill(3))
    o = set(str(row['open_code']).zfill(3))
    return len(s & o)

df['match_count'] = df.apply(match_count, axis=1)

# === 位置命中 ===
def match_pos(row):
    sim = str(row['sim_test_code']).zfill(3)
    open_ = str(row['open_code']).zfill(3)
    return sum([1 for i in range(3) if sim[i] == open_[i]])

df['match_pos_count'] = df.apply(match_pos, axis=1)

# === 独胆：从下期百位视为独胆 ===
df['single_digit'] = df['open_digit_1'].shift(-1)

# === 三位号 ===
df['sim_digit_1'] = df['sim_test_code'].astype(str).str.zfill(3).str[0].astype(int)
df['sim_digit_2'] = df['sim_test_code'].astype(str).str.zfill(3).str[1].astype(int)
df['sim_digit_3'] = df['sim_test_code'].astype(str).str.zfill(3).str[2].astype(int)

# === Rolling 热度 ===
df['single_hot_5'] = df['single_digit'].rolling(5, min_periods=1).apply(lambda x: Counter(x).most_common(1)[0][1])
df['single_hot_3'] = df['single_digit'].rolling(3, min_periods=1).apply(lambda x: Counter(x).most_common(1)[0][1])

# === 过滤最后一行 ===
df = df.dropna().reset_index(drop=True)

# === 保存 ===
df.to_csv(LABELS_PATH, index=False)
print(f"✅ 已保存: {LABELS_PATH}")
print(df.tail(2))
