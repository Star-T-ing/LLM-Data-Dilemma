import json
import re
import numpy as np
import torch
from lmppl import MaskedLM 

# --- 1. 配置---
MODEL_NAME = '/home/remote1/lvshuyang/Models/hfl/chinese-bert-wwm-ext'  # 本地模型路径
INPUT_FILE = 'generated_predictions_augmented.jsonl' # 待评估的文件
BATCH_SIZE = 16     # 批次大小
PPL_THRESHOLD = 500 # PPL阈值，来识别和过滤异常值
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 句子拆分函数---
def split_sentences(text):
    if not text:
        return []
    sentences = re.split(r'([。！？；])', text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + sentences[i+1]
        if sentence.strip():
            result.append(sentence.strip())
    if sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result

# --- 3. 主评估流程 ---
def evaluate_fluency(model_name, file_path, device, batch_size):
    print(f"正在从本地路径加载MaskedLM模型: {model_name} 到设备: {device}")
    
    try:
        scorer = MaskedLM(
            model=model_name, 
            max_length=512 # 最大token长度
        )
    except Exception as e:
        print(f"加载模型失败，请检查路径是否正确。错误: {e}")
        return None

    print("模型加载完毕。开始处理文件...")
    
    results = []
    all_ppl_scores = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 对每一行的数据进行评测
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            # 将 prompt 和 predict 拼接
            predict_text = data.get('prompt', '') + data.get('predict', '')

            if not predict_text.strip():
                print(f"处理第 {i + 1} 行时发生错误: {'预测文本为空，已跳过。'}")
                continue

            sentences = split_sentences(predict_text)

            if not sentences:
                print(f"处理第 {i + 1} 行时发生错误: {'无法拆分出有效句子，已跳过。'}")
                continue

            # 计算PPL
            ppl_scores = scorer.get_perplexity(sentences, batch_size=batch_size)

            # 检查是否存在异常PPL值，如果存在则跳过该样本
            if any(score > PPL_THRESHOLD for score in ppl_scores):
                print(f"警告：第 {i + 1} 行包含PPL异常值 (>{PPL_THRESHOLD})，已忽略该样本。异常值: {max(ppl_scores):.2f}")
                continue 

            avg_ppl = np.mean(ppl_scores)
            all_ppl_scores.append(avg_ppl)
            
            # 记录结果
            results.append({
                "line_no": i + 1,
                "sentence_count": len(sentences),
                "ppl_scores": ppl_scores,
                "average_ppl": avg_ppl
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(lines):
                print(f"已处理 {i + 1}/{len(lines)} 条数据，当前平均PPL: {np.mean(all_ppl_scores):.4f}")

        except json.JSONDecodeError:
            print(f"警告：第 {i + 1} 行不是有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"处理第 {i + 1} 行时发生错误: {e}")

    total_avg_ppl = np.mean(all_ppl_scores) if all_ppl_scores else 0

    # 将最终总结 append 到结果列表中
    results.append({
                "test_file": file_path,
                "total_samples": len(lines),
                "total_valid_samples": len(all_ppl_scores),
                "total_average_ppl": total_avg_ppl
            })
    
    # 打印最终摘要
    print("\n" + "="*50)
    print("              Fluency Score Summary (PPL)")
    print("="*50)
    print(f"评估文件: {file_path}")
    print(f"有效文本数量: {len(all_ppl_scores)}")
    print("-" * 50)
    print(f"  - Average Perplexity (文本流畅度↓): {total_avg_ppl:.4f}")
    print("="*50)
    
    return results, total_avg_ppl

# --- 4. 运行评估 ---
if __name__ == "__main__":
    # 执行评估
    evaluation_results, total_ppl = evaluate_fluency(MODEL_NAME, INPUT_FILE, DEVICE, BATCH_SIZE)
    
    # 保存详细的评估结果到文件
    if evaluation_results:
        OUTPUT_FILE = f"fluency_results_{INPUT_FILE}"
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(evaluation_results, outfile, ensure_ascii=False, indent=4)
        print(f"详细评估结果已保存到 {OUTPUT_FILE}")
        
        