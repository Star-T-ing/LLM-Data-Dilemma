import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 全局配置 (Global Configuration)
# ==========================================================
THUCNEWS_ROOT = r"E:\Graduate_doc\school_life\THUCNEWS_data\THUCNews1\THUCNews1"

TARGET_CATEGORIES = ["财经", "房产","教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]


# ==========================================================
# 函数 1: 数据读取 (Data Loading)
# ==========================================================
def read_thucnews(root_dir, target_categories):
    data = []

    # 外层循环：遍历分类
    for category in target_categories:
        category_dir = os.path.join(root_dir, category)
        if not os.path.exists(category_dir):
            print(f"⚠️ 跳过：分类文件夹不存在 {category}")
            continue

        # 获取该分类下所有文件名列表
        filenames = [f for f in os.listdir(category_dir) if f.endswith(".txt")]

        for filename in tqdm(filenames, desc=f"正在处理 [{category}]", unit="doc"):
            file_path = os.path.join(category_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # 简单的拆分逻辑
                if "\t" in content:
                    title, text = content.split("\t", maxsplit=1)
                else:
                    title = content[:30].strip()
                    text = content[30:].strip()

                data.append({
                    "category": category,
                    "doc_id": filename.replace(".txt", ""),
                    "title": title,
                    "content": text
                })
            except Exception as e:
                print(f"\n❌ 读取失败 {file_path}：{e}")
                continue

    df = pd.DataFrame(data)
    print("\n" + "=" * 30)
    print(f"✅ 读取完成！")
    print(f"总样本数：{len(df)}")
    print(f"包含分类：{df['category'].unique()}")
    print("=" * 30)
    return df


# ==========================================================
# 函数 2: 数据分析 (EDA 和质量检查) (Enhanced version with cleaning)
# ==========================================================
def perform_eda_and_quality_check_enhanced(df: pd.DataFrame):
    """
    对已加载的 THUCNews DataFrame 进行探索性数据分析和质量检查（已修复中文乱码，包含清洗步骤）。
    """

    # 清理特殊字符（全角空格，换行符，不间断空格，制表符）
    def clean_text(text):
        if isinstance(text, str):
            text = text.replace("\u3000", "")  # 替换全角空格
            text = text.replace("\n", " ")  # 替换换行符
            text = text.replace("\xa0", "")  # 替换不间断空格
            text = text.replace("\t", " ")  # 替换制表符
        return text

    # 应用文本清洗到标题和正文
    df["title"] = df["title"].apply(clean_text)
    df["content"] = df["content"].apply(clean_text)

    # 长度过滤：删除正文长度小于50个字符的样本
    df = df[df["content"].apply(len) >= 50]

    # ==========================================================
    # 基本信息与长度计算
    # ==========================================================
    df["title_len"] = df["title"].fillna("").apply(len)
    df["content_len"] = df["content"].fillna("").apply(len)
    category_counts = df["category"].value_counts()

    # 设置字体以避免中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 输出基本信息和类别分布
    print("=" * 50)
    print(f"数据集形状 (行数, 列数)：{df.shape}")
    print("分类分布（样本量）：")
    print(category_counts)

    # 绘制类别分布图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('分类样本数量分布', fontsize=16)
    plt.xlabel('新闻类别', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 绘制正文长度分布图
    plt.figure(figsize=(12, 6))
    sns.histplot(df["content_len"], bins=50, kde=True,
                 color='skyblue', edgecolor='black', line_kws={'linewidth': 3})
    plt.title('正文长度分布直方图', fontsize=16)
    plt.xlabel('正文长度 (字符数)', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 检查缺失值
    print("\n缺失值统计：")
    print(df.isnull().sum())

    # 输出文本长度统计摘要
    print("\n### 标题长度统计：")
    print(df["title_len"].describe())

    print("\n### 正文长度统计：")
    print(df["content_len"].describe())

    # 重复值检查
    print("\n重复值检查（精确去重）")
    df = find_exact_duplicates(df)

    return df




# ==========================================================
# 主执行块 (Main Execution Block)
# ==========================================================
if __name__ == "__main__":
    # 1. 数据读取：加载原始数据
    df = read_thucnews(THUCNEWS_ROOT, TARGET_CATEGORIES)

    # 2. EDA 和质量检查：分析数据并输出图表
    df = perform_eda_and_quality_check_enhanced(df)


    print("\n数据处理完毕")
