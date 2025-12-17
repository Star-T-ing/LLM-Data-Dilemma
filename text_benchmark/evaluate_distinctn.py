import json
import re
import jieba
import numpy as np
from tqdm import tqdm

# --- 1. 配置 ---
INPUT_FILE = 'generated_predictions_augmented.jsonl'  # 待评估的文件

# --- 2. 辅助函数 ---
def clean_and_tokenize(text: str) -> list:
    """组合文本清洗和分词的流程。"""
    # 1. 清洗文本：移除标点和特殊字符
    punctuation_pattern = r"[\s,.!?;:\"“”、，。《》（）——+-=【】*&^%$#@!<>~`'·]+"
    cleaned_text = re.sub(punctuation_pattern, "", text)
    
    # 2. 使用jieba进行分词
    if not cleaned_text:
        return []
    return jieba.lcut(cleaned_text)

def calculate_distinct_metrics_for_sample(tokens: list) -> tuple:
    """为单条文本的token列表计算Distinct-1和Distinct-2。"""
    if not tokens:
        return 0.0, 0.0

    # 计算 Distinct-1
    distinct_1 = 0.0
    # n-gram在这里就是单个词
    ngrams_1 = [tuple([token]) for token in tokens]
    if ngrams_1:
        distinct_1 = len(set(ngrams_1)) / len(ngrams_1)
        
    # 计算 Distinct-2
    distinct_2 = 0.0
    if len(tokens) >= 2:
        ngrams_2 = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
        if ngrams_2:
            distinct_2 = len(set(ngrams_2)) / len(ngrams_2)
            
    return distinct_1, distinct_2

# --- 3. 主评估函数 ---
def evaluate_diversity(file_path: str):
    """
    计算文件中每条生成文本的多样性，并将所有样本的多样性分数取平均作为最终结果。
    """
    print(f"开始处理文件: {file_path}")
    
    # 用于存储每条数据的详细结果
    per_line_results = []
    
    # 用于计算最终平均分：存储每条样本的d1和d2分数
    all_d1_scores = []
    all_d2_scores = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines, desc=f"Calculating Diversity for {file_path}")):
        try:
            data = json.loads(line)
            predict_text = data.get('predict', '')
            
            tokens = clean_and_tokenize(predict_text)
            
            if not tokens:
                continue
            
            # --- 1. 计算单条样本的多样性分数 ---
            d1_score, d2_score = calculate_distinct_metrics_for_sample(tokens)
            
            # --- 2. 存储每条样本的详细结果 ---
            per_line_results.append({
                "line_no": i + 1,
                "token_count": len(tokens),
                "distinct_1": round(d1_score, 4),
                "distinct_2": round(d2_score, 4)
            })
            
            # --- 3. 收集分数用于最后计算平均值 ---
            all_d1_scores.append(d1_score)
            all_d2_scores.append(d2_score)

        except (json.JSONDecodeError, AttributeError):
            tqdm.write(f"警告: 第 {i + 1} 行不是有效的JSON格式或内容，已跳过。")
            continue
            
    # --- 结果汇总 ---
    
    # 计算所有样本分数的平均值
    final_avg_d1 = np.mean(all_d1_scores) if all_d1_scores else 0
    final_avg_d2 = np.mean(all_d2_scores) if all_d2_scores else 0

    # 准备最终的概要报告
    summary_results = {
        "file_path": file_path,
        "total_valid_texts": len(per_line_results),
        "average_distinct_1": f"{final_avg_d1:.4f}",
        "average_distinct_2": f"{final_avg_d2:.4f}"
    }
    
    return per_line_results, summary_results

# --- 4. 运行评估 ---
if __name__ == "__main__":
    # 执行评估
    detailed_results, final_summary = evaluate_diversity(INPUT_FILE)
    
    # 保存详细的评估结果到文件
    if detailed_results:
        detailed_results.append(final_summary)  # 在详细结果中添加最终概要
        OUTPUT_FILE = f"diversity_results_{INPUT_FILE}"
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)
        print(f"\n详细的评估结果已保存到: {OUTPUT_FILE}")

    # 打印最终的概要报告
    print("\n" + "="*50)
    print("           Average Diversity Score Summary")
    print("="*50)
    print(f"评估文件: {final_summary['file_path']}")
    print(f"有效文本数量: {final_summary['total_valid_texts']}")
    print("-" * 50)
    print(f"  - Average Distinct-1 (词汇丰富度): {final_summary['average_distinct_1']}")
    print(f"  - Average Distinct-2 (短语丰富度): {final_summary['average_distinct_2']}")
    print("="*50)
    
    