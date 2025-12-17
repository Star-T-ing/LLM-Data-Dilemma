import os
import json
import time
import re
from zai import ZhipuAiClient
from tqdm import tqdm
import numpy as np
import concurrent.futures

# --- 1. 配置 ---
# API配置
API_KEY = "2dce28f531864ae6b382bfd4d2dd3828.dnK3H2D5cZYf6DoQ"
MODEL_NAME = "glm-4.5-flash"
TEMPERATURE = 0.7

# 文件名配置
INPUT_FILE = 'generated_predictions_augmented.jsonl'  # 待评估文件名
OUTPUT_FILE = f"glm4eval_results_{INPUT_FILE.replace('.jsonl', '.json')}"
ERROR_LOG_FILE = f"glm4eval_errors_{INPUT_FILE.replace('.jsonl', '.log')}"

# 并发与数据量配置
CONCURRENCY_LEVEL = 2  # 设置并发请求数
MAX_SAMPLES_TO_EVALUATE = None # 设置为 None 则评估所有数据

# API调用配置
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120
RATE_LIMIT_DELAY = 1

# --- 2. Prompt模板---
PROMPT_TEMPLATE = """
### 角色 ###
你是一位资深的、要求严格的新闻总编，负责对AI记者生成的稿件进行终审。

### 任务 ###
你的任务是根据一套清晰、独立的评估标准，对AI模型续写的新闻文本进行1-5分的整数评分。请严格遵循每一个指标的定义，确保评分的公正性和准确性。

### 核心评估指标与标准 ###

1.  **连贯性 (Coherence) - [1-5分]**:
    *   **评估对象**: 【模型续写】内部的逻辑流程。
    *   **标准**: 1分代表内部逻辑混乱或矛盾，5分代表内部逻辑严密、行文流畅。

2.  **信息量 (Informativeness) - [1-5分]**:
    *   **评估对象**: 【模型续写】提供的新内容价值。
    *   **标准**: 1分代表内容空洞、全是重复或废话，5分代表提供了具体、有价值的信息。

3.  **相关性 (Relevance) - [1-5分]**:
    *   **评估对象**: 【模型续写】与【新闻开头】的衔接程度。
    *   **标准**: 1分代表与开头完全无关、严重跑题，5分代表与开头无缝衔接、是主题的合理延续。

4.  **新闻风格 (News Style) - [1-5分]**:
    *   **评估对象**: 【模型续写】的语言风格是否符合新闻文体。
    *   **标准**: 1分代表风格完全错误（如口语化、情绪化），5分代表语言清晰、明确、客观，完全符合新闻文风。

### 输入 ###
【新闻开头】:
\"\"\"
{prompt_text}
\"\"\"

【模型续写】:
\"\"\"
{predict_text}
\"\"\"

### 输出要求 ###
请先在脑海中进行一步步的分析（思维链），然后将最终结果以一个严格的JSON对象格式输出，不要包含任何JSON格式之外的额外解释。JSON结构如下：

{{
  "analysis": "这里是你对续写文本的简要分析，说明你打分的原因。",
  "coherence_score": <1-5之间的整数>,
  "informativeness_score": <1-5之间的整数>,
  "relevance_score": <1-5之间的整数>,
  "news_style_score": <1-5之间的整数>
}}
"""

# --- 3. 辅助函数---
def parse_json_from_response(response_text):
    if not response_text: return None
    try: return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except json.JSONDecodeError: return None
    return None

def log_error(line_num, error_type, content):
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Line {line_num}: [{error_type}]\nContent: {content}\n\n")

def save_final_results(summary, detailed_results, file_path):
    final_data = {"summary": summary, "detailed_results": detailed_results}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

def get_summary(results):
    if not results: return {}
    score_keys = [key for key in results[0].keys() if key.endswith('_score')]
    summary = {
        "evaluation_file": INPUT_FILE, "model_used": MODEL_NAME,
        "total_valid_samples": len(results), "average_scores": {}
    }
    for key in score_keys:
        scores = [r.get(key) for r in results if isinstance(r.get(key), int)]
        if scores: summary["average_scores"][key] = round(np.mean(scores), 4)
    return summary

# --- 4. 处理单条数据的函数 (用于并发) ---
def process_single_item(data, line_num, client):
    """处理单条数据，包括API调用和重试逻辑。"""
    try:
        prompt_text = data.get('prompt', '')
        predict_text = data.get('predict', '')

        if not predict_text.strip():
            log_error(line_num, "Skipped: Predict text is empty.", "")
            return None
        
        formatted_prompt = PROMPT_TEMPLATE.format(prompt_text=prompt_text, predict_text=predict_text)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME, 
                    messages=[{"role": "user", "content": formatted_prompt}],
                    response_format={"type": "json_object"}, # 结构化输出
                    thinking={"type": "disabled"},  # 禁止深度思考模式,
                    temperature=TEMPERATURE, timeout=REQUEST_TIMEOUT, max_tokens=3072
                )
                response_content = response.choices[0].message.content
                response_json = parse_json_from_response(response_content)
                if response_json:
                    response_json['original_line_num'] = data.get('line_num', line_num)
                    return response_json
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
                else:
                    log_error(line_num, "API Error after max retries", str(e))
        
        log_error(line_num, "Failed to get valid JSON response", response_content if 'response_content' in locals() else 'No response')
        return None

    except Exception as e:
        log_error(line_num, "Unexpected error in main processing loop", str(e))
        return None

# --- 5. 主评估流程 (并发模式) ---
def main():
    try:
        client = ZhipuAiClient(api_key=API_KEY)
    except Exception as e:
        print(f"初始化 ZhipuAiClient 失败: {e}")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            if INPUT_FILE.endswith('.jsonl'):
                lines = f.readlines()
                input_data = [json.loads(line) for line in lines if line.strip()]
            else:
                full_data = json.load(f)
                if isinstance(full_data, list): input_data = full_data
                elif "detailed_results" in full_data and isinstance(full_data["detailed_results"], list): input_data = full_data["detailed_results"]
                elif "predict" in full_data[0]: input_data = full_data
                else: raise ValueError("在JSON文件中找不到可评估的数据列表")
    except Exception as e:
        print(f"读取或解析输入文件 {INPUT_FILE} 时失败: {e}")
        return

    # 根据配置对数据进行切片
    if MAX_SAMPLES_TO_EVALUATE is not None:
        input_data = input_data[:MAX_SAMPLES_TO_EVALUATE]
    
    print(f"成功提取 {len(input_data)} 条数据进行评估 (并发数: {CONCURRENCY_LEVEL})。")

    detailed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY_LEVEL) as executor:
        # 提交所有任务
        future_to_line_num = {executor.submit(process_single_item, data, i + 1, client): i + 1 for i, data in enumerate(input_data)}
        
        # 使用tqdm显示并发处理进度
        for future in tqdm(concurrent.futures.as_completed(future_to_line_num), total=len(input_data), desc=f"Evaluating with {MODEL_NAME}"):
            result = future.result()
            if result:
                detailed_results.append(result)

    print("\n评估流程全部完成。正在生成最终报告...")

    # 排序结果，确保输出文件顺序与输入文件一致
    detailed_results.sort(key=lambda x: x['original_line_num'])

    final_summary = get_summary(detailed_results)
    save_final_results(final_summary, detailed_results, OUTPUT_FILE)

    print(f"评估报告已成功保存到: {OUTPUT_FILE}")
    print("\n" + "="*50)
    print(f"           {MODEL_NAME} Evaluation Summary")
    print("="*50)
    for key, value in final_summary.items():
        if key == "average_scores":
            print("-" * 50)
            for metric, score in value.items():
                print(f"  - Average {metric}: {score}")
        else:
            print(f"{key}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()
    
    