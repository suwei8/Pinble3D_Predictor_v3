import sys
import os
import subprocess
import pandas as pd
import shutil
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
RESULT_PATH = os.path.join(BASE_DIR, "data", "next_predict_result.csv")
JSON_RESULT_PATH = os.path.join(BASE_DIR, "data", "predict_result.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

COLLECTOR = os.path.join(BASE_DIR, "collector", "Lottery_3d_local.py")
FEATURE_GEN = os.path.join(BASE_DIR, "predictor", "feature_generator.py")
TRAIN_REAL = os.path.join(BASE_DIR, "predictor", "train_real.py")
PREDICT_REAL = os.path.join(BASE_DIR, "predictor", "predict_tft.py")

import collector.Lottery_3d_local as collector

def run(cmd):
    code = subprocess.call(cmd, shell=True)
    if code != 0:
        raise RuntimeError(f"执行失败: {cmd}")

def run_and_capture(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"执行失败: {cmd}")
    return result.stdout

def is_models_empty():
    return not os.path.exists(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0

def main():
    # 初始化结果文件
    if not os.path.exists(RESULT_PATH):
        pd.DataFrame(columns=[
            "issue", "pred_dandan", "true_dandan", "hit",
            "pred_digits", "true_digits", "pos_hit"
        ]).to_csv(RESULT_PATH, index=False)

    while True:
        next_truth = collector.get_next_truth()
        if next_truth is None:
            print("✅ 所有期号已补齐，回测完成")
            break

        print(f"🚀 本轮目标：预测期号 {int(next_truth['issue'])}")

        # === 特征生成 ===
        run(f"python \"{FEATURE_GEN}\"")

        if is_models_empty():
            print("🔁 模型目录为空 ➜ 做全量训练")
            run(f"python \"{TRAIN_REAL}\" --mode full")
        else:
            print("🔁 模型目录存在 ➜ 做增量训练")
            run(f"python \"{TRAIN_REAL}\" --mode incremental")

        # === 预测 ===
        run(f"python \"{PREDICT_REAL}\"")

        # === 读取 JSON 预测结果 ===
        with open(JSON_RESULT_PATH, "r", encoding="utf-8") as f:
            predict_data = json.load(f)

        pred_dandan = predict_data["pred_dandan"]
        pred_digits = predict_data["pred_digits"]

        true_dandan = int(str(next_truth["open_code"]).zfill(3)[0])
        true_digits = [int(x) for x in str(next_truth["open_code"]).zfill(3)]

        hit = "✔️" if pred_dandan == true_dandan else "❌"
        pos_hit = sum([a == b for a, b in zip(pred_digits, true_digits)])

        # === 结果保存到 CSV ===
        df_result = pd.read_csv(RESULT_PATH)
        df_result = pd.concat([df_result, pd.DataFrame([{
            "issue": next_truth["issue"],
            "pred_dandan": pred_dandan,
            "true_dandan": true_dandan,
            "hit": hit,
            "pred_digits": pred_digits,
            "true_digits": true_digits,
            "pos_hit": pos_hit
        }])])
        df_result.to_csv(RESULT_PATH, index=False)

        print(f"✅ 期号 {next_truth['issue']} ➜ 命中记录已写入")

        # === 真值追加到历史 ===
        df_cur = pd.read_csv(HISTORY_PATH)
        df_new = pd.concat([df_cur, next_truth.to_frame().T], ignore_index=True)
        df_new.to_csv(HISTORY_PATH, index=False)

    print("🎉 真回测闭环结束")

if __name__ == "__main__":
    main()
