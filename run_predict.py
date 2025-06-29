# run_pipeline.py
import os
import subprocess
import sys

# ========== 📌 路径 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON = sys.executable  # 当前虚拟环境 Python

def run(cmd, cwd=None):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, cwd=cwd or BASE_DIR, shell=True)
    if result.returncode != 0:
        print(f"❌ Step failed: {cmd}")
        sys.exit(result.returncode)

# ========== 1️⃣ 采集最新数据 ==========
print("\n=== [Step 1] 采集最新数据 ===")
run(f"{PYTHON} collector/Lottery_3d.py")

# ========== 2️⃣ 生成新标签（这里假设你有 feature_generator.py）==========
print("\n=== [Step 2] 生成新标签 ===")
run(f"{PYTHON} predictor/feature_generator.py")

# ========== 3️⃣ 增量训练模型 ==========
print("\n=== [Step 3] 增量训练 ===")
run(f"{PYTHON} predictor/train_real_incremental.py")

# ========== 4️⃣ 预测最新一期 ==========
print("\n=== [Step 4] 预测最新一期 ===")
run(f"{PYTHON} predictor/predict_tft.py")

print("\n✅ 全流程已完成！")
