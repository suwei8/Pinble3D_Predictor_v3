# collector/Lottery_3d_local.py

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history_all.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")

def get_next_truth():
    df_all = pd.read_csv(ALL_PATH).sort_values("issue").reset_index(drop=True)
    df_cur = pd.read_csv(HISTORY_PATH).sort_values("issue").reset_index(drop=True)

    last_issue = df_cur["issue"].max()
    next_rows = df_all[df_all["issue"] > last_issue]

    if next_rows.empty:
        print("✅ 无新期号可用于核对")
        return None

    next_row = next_rows.iloc[0]
    print(f"✅ 下一期真值: {next_row['issue']}")
    return next_row

if __name__ == "__main__":
    get_next_truth()
