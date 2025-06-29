# predictor/feature_generator.py

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_features.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")

# ✅ 加载历史数据
df = pd.read_csv(HISTORY_PATH).dropna().reset_index(drop=True)
print(f"✅ 已加载历史 {len(df)} 条")

# === 🔑 新增目标列 ===
df["next_open_code"] = df["open_code"].shift(-1)

# === 🔑 新增示例特征（你可扩展）
df["sim_sum_val"] = df["sim_code"].astype(str).apply(lambda x: sum(int(c) for c in x))
df["open_sum_val"] = df["open_code"].astype(str).apply(lambda x: sum(int(c) for c in x))

# === 你可以在这里插入更多特征列 ===

# ✅ 生成训练集（去掉最后一个 NaN）
df_labels = df.dropna(subset=["next_open_code"]).copy()
df_labels.to_csv(LABELS_PATH, index=False)
print(f"✅ 已保存标签集 {len(df_labels)} 条 -> {LABELS_PATH}")

# ✅ 保留最新一行用于未来预测
df_features = df.copy()
df_features.to_csv(FEATURES_PATH, index=False)
print(f"✅ 已保存特征集 {len(df_features)} 条 -> {FEATURES_PATH}")

print(f"🎯 你现在可以用 {df_features.iloc[-1]['issue']} 期去预测下一期！")
