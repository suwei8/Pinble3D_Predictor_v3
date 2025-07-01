# run_pipeline.py
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def run(cmd, cwd=None):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, cwd=cwd or BASE_DIR, shell=True)
    if result.returncode != 0:
        print(f"❌ Step failed: {cmd}")
        sys.exit(result.returncode)

print("\n=== [Step 1] 采集最新数据 ===")
run(f"{PYTHON} collector/Lottery_3d.py")

print("\n=== [Step 2] 生成新标签 ===")
run(f"{PYTHON} predictor/feature_generator.py")

print("\n=== [Step 3] 增量训练 ===")
run(f"{PYTHON} predictor/train_real.py --mode incremental")

print("\n=== [Step 4] 预测最新一期 ===")
run(f"{PYTHON} predictor/predict_tft.py")

print("\n=== [Step 5] 微信推送提醒 ===")
run(f"{PYTHON} notifier/wechat_notify.py")

print("\n✅ 全流程已完成，已推送微信 ✅")
