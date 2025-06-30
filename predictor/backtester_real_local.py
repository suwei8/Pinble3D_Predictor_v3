# predictor/backtester_real_local.py

import os
import subprocess
import pandas as pd
import shutil
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
HISTORY_ALL_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history_all.csv")
RESULT_PATH = os.path.join(BASE_DIR, "data", "next_predict_result.csv")
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

def clear_models():
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    if not os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            f.write("issue|pred_dandan|true_dandan|hit|pred_digits|true_digits|pos_hit\n")

    while True:
        # 获取下一期真值
        next_truth = collector.get_next_truth()
        if next_truth is None:
            print("✅ 所有期号已补齐，回测完成")
            break

        print(f"🚀 本轮目标：预测期号 {int(next_truth['issue'])}")

        clear_models()

        # 重新生成特征，用历史全量做训练
        run(f"python \"{FEATURE_GEN}\"")
        run(f"python \"{TRAIN_REAL}\"")

        # === 捕获 predict_tft.py 输出
        stdout = run_and_capture(f"python \"{PREDICT_REAL}\"")
        print(stdout)

        # === 正则提取预测值
        pred_dandan_match = re.search(r"预测独胆:\s*(\d+)", stdout)
        pred_seq_match = re.search(r"预测试机号3位:\s*\[([^\]]+)\]", stdout)

        if not (pred_dandan_match and pred_seq_match):
            raise ValueError("预测输出格式异常")

        pred_dandan = int(pred_dandan_match.group(1))
        pred_seq_digits = [int(x) for x in pred_seq_match.group(1).split(",")]

        # 用下一期真值做对比
        true_dandan = int(str(next_truth["open_code"]).zfill(3)[0])
        true_digits = [int(x) for x in str(next_truth["open_code"]).zfill(3)]

        hit = "✔️" if pred_dandan == true_dandan else "❌"
        pos_hit = sum([a == b for a, b in zip(pred_seq_digits, true_digits)])

        with open(RESULT_PATH, "a", encoding="utf-8") as f:
            f.write(f"{next_truth['issue']}|{pred_dandan}|{true_dandan}|{hit}|{pred_seq_digits}|{true_digits}|{pos_hit}\n")

        print(f"✅ 期号 {next_truth['issue']} ➜ 命中记录已写入")

        # 追加真值到 history.csv
        df_cur = pd.read_csv(HISTORY_PATH)
        df_new = pd.concat([df_cur, next_truth.to_frame().T], ignore_index=True)
        df_new.to_csv(HISTORY_PATH, index=False)
        print(f"✅ 已把 {next_truth['issue']} 真值追加到历史")

    print("🎉 真回测闭环结束")

if __name__ == "__main__":
    main()
