import os
import pandas as pd
import random
from shutil import copyfile
from tqdm import tqdm

# ==========================================================
# 配置路径
# ==========================================================
THUCNEWS_ROOT = r"E:\Graduate_doc\school_life\THUCNEWS_data\THUCNews\THUCNews1"
TARGET_CATEGORIES = ["财经", "房产","教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
OUTPUT_DIR = "random_data"
NUM_SAMPLES = 50000  # 抽取的样本数量

# ==========================================================
# 创建目标文件夹 (如果不存在)
# ==========================================================
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ==========================================================
# 随机抽取5万条数据
# ==========================================================
def extract_random_data(root_dir, categories, num_samples, output_dir):
    all_files = []

    # 遍历每个分类文件夹，收集所有文件的路径
    for category in categories:
        category_dir = os.path.join(root_dir, category)
        if not os.path.exists(category_dir):
            print(f"⚠️ 跳过：分类文件夹不存在 {category}")
            continue

        filenames = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith(".txt")]
        all_files.extend(filenames)

    # 随机选择5万条文件
    random_files = random.sample(all_files, num_samples)

    # 将选中的文件复制到目标文件夹中
    for idx, file_path in tqdm(enumerate(random_files), desc="正在复制文件", unit="file"):
        try:
            # 复制文件到目标文件夹
            target_path = os.path.join(output_dir, f"sample_{idx + 1}.txt")
            copyfile(file_path, target_path)
        except Exception as e:
            print(f"❌ 复制文件失败: {file_path} -> {e}")

    print(f"\n✅ 随机抽取并复制了 {num_samples} 条数据到 '{output_dir}' 文件夹中。")


# 执行抽取
extract_random_data(THUCNEWS_ROOT, TARGET_CATEGORIES, NUM_SAMPLES, OUTPUT_DIR)
