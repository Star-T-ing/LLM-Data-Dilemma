import gradio as gr
import json
import os
from collections import Counter

# --- 1. 全局配置 ---
# 配置两个需要对比的模型输出文件
FILE_A = "pred/generated_predictions_cleaned.jsonl"    # 模型 A (例如：增强前)
FILE_B = "generated_predictions_augmented.jsonl"  # 模型 B (例如：增强后)

# 自动生成评分文件名
SCORES_FILE = f"{os.path.splitext(FILE_A)[0]}_vs_{os.path.splitext(os.path.basename(FILE_B))[0]}_scores.json"

# --- 2. 数据加载与状态管理 ---
merged_data = []
scores = {}

def load_data():
    """从两个文件中加载数据并将它们按行合并。"""
    global merged_data
    merged_data = []
    data_a, data_b = [], []
    try:
        with open(FILE_A, 'r', encoding='utf-8') as f:
            for line in f: data_a.append(json.loads(line.strip()))
    except FileNotFoundError: return 0
    try:
        with open(FILE_B, 'r', encoding='utf-8') as f:
            for line in f: data_b.append(json.loads(line.strip()))
    except FileNotFoundError: return 0

    min_len = min(len(data_a), len(data_b))
    for i in range(min_len):
        merged_data.append({
            "prompt": data_a[i].get("prompt", "N/A"),
            "predict_A": data_a[i].get("predict", "N/A"),
            "predict_B": data_b[i].get("predict", "N/A")
        })
    print(f"数据加载完成，共 {len(merged_data)} 条可供评测。")
    return len(merged_data)

def load_scores():
    """加载评分，兼容新旧两种格式。"""
    global scores
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容新格式 (带summary) 和旧格式 (只有scores)
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
            else:
                scores = data
            scores = {str(k): v for k, v in scores.items()}
        print(f"成功加载 {len(scores)} 条已有评分。")
    else:
        scores = {}

# --- 3. 核心交互函数 ---
def create_colored_html(prompt, predict):
    """生成带有颜色区分的HTML文本。"""
    return (
        f"<div style='font-family: sans-serif; line-height: 1.6; border: 1px solid #E5E7EB; border-radius: 5px; padding: 10px;'>"
        f"<span style='background-color: #EBF5FB; padding: 2px 4px; border-radius: 3px;'>{prompt}</span>"
        f"<span style='background-color: #E8F8F5; padding: 2px 4px; border-radius: 3px;'>{predict}</span>"
        f"</div>"
    )

def get_sample(index):
    """获取指定索引的数据并准备显示。"""
    index = int(index)
    sample = merged_data[index]
    html_a = create_colored_html(sample.get("prompt"), sample.get("predict_A"))
    html_b = create_colored_html(sample.get("prompt"), sample.get("predict_B"))
    current_choice = scores.get(str(index))
    rating_status = f"状态: 已评分 ({current_choice})" if current_choice else "状态: 尚未评分"
    status = f"正在查看第 {index + 1} / {len(merged_data)} 条"
    return html_a, html_b, current_choice, status, rating_status

def get_score_stats(total_samples_count):
    """计算核心统计数据并返回一个字典。"""
    rated_count = len(scores)
    if rated_count == 0:
        return {"rated_count": 0, "win_a": 0, "win_b": 0, "tie": 0, "win_a_p": 0, "win_b_p": 0, "tie_p": 0}
    
    counts = Counter(scores.values())
    win_a = counts.get("模型 A 更好", 0)
    win_b = counts.get("模型 B 更好", 0)
    tie = counts.get("平局 / 质量相当", 0)
    
    return {
        "rated_count": rated_count,
        "total_count": total_samples_count,
        "win_a": win_a, "win_b": win_b, "tie": tie,
        "win_a_p": f"{win_a/rated_count:.2%}",
        "win_b_p": f"{win_b/rated_count:.2%}",
        "tie_p": f"{tie/rated_count:.2%}",
    }

def generate_analysis_text(stats):
    """根据统计数据字典生成供显示的文本。"""
    if stats['rated_count'] == 0:
        return "尚未对任何样本进行评分。"
    
    return (
        f"已评分/总量: {stats['rated_count']} / {stats['total_count']}\n"
        f"模型 A 胜: {stats['win_a']} ({stats['win_a_p']})\n"
        f"模型 B 胜: {stats['win_b']} ({stats['win_b_p']})\n"
        f"平局: {stats['tie']} ({stats['tie_p']})"
    )

def save_scores_and_summary(total_samples_count):
    """【核心修改】保存包含概要和详细评分的完整JSON文件。"""
    stats = get_score_stats(total_samples_count)
    summary = {
        "model_a_file": FILE_A,
        "model_b_file": FILE_B,
        "rated_samples": stats['rated_count'],
        "total_samples": stats['total_count'],
        "score_distribution": {
            "model_a_wins": stats['win_a'],
            "model_b_wins": stats['win_b'],
            "ties": stats['tie']
        },
        "win_percentages": {
            "model_a": stats['win_a_p'],
            "model_b": stats['win_b_p'],
            "ties": stats['tie_p']
        }
    }
    final_data = {"summary": summary, "scores": scores}
    with open(SCORES_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

def update_score_and_analysis(index, choice, total_samples_count):
    """【核心修改】评分后，立即更新状态、保存并刷新分析结果。"""
    if index is None or choice is None:
        return "状态未知", "分析结果待更新"
    
    index = int(index)
    scores[str(index)] = choice
    save_scores_and_summary(total_samples_count)
    
    rating_status = f"状态: 已评分 ({choice})"
    stats = get_score_stats(total_samples_count)
    analysis_text = generate_analysis_text(stats)
    
    return rating_status, analysis_text

# --- 4. Gradio 界面构建 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title= "生成文本质量对比评测工具") as demo:
    total_samples = gr.State(value=0)
    current_index = gr.State(value=0)

    gr.Markdown("# 生成文本质量对比评测工具")
    gr.Markdown(f"请对比 **模型 A (`{os.path.basename(FILE_A)}`)** 和 **模型 B (`{os.path.basename(FILE_B)}`)** 的输出。<br>共享的新闻开头以<span style='background-color: #EBF5FB;'>浅蓝色</span>高亮，AI续写部分以<span style='background-color: #E8F8F5;'>浅绿色</span>高亮。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 模型 A 输出")
            prediction_a_html = gr.HTML()
        with gr.Column(scale=1):
            gr.Markdown("### 模型 B 输出")
            prediction_b_html = gr.HTML()
    
    gr.Markdown("---")

    score_radio = gr.Radio(
        ["模型 A 更好", "模型 B 更好", "平局 / 质量相当"], 
        label="哪个续写更好？",
        info="请综合考虑流畅度、连贯性、信息量和相关性。"
    )

    with gr.Row(equal_height=True):
        with gr.Column(min_width=250):
            gr.Markdown("#### 导航")
            with gr.Row():
                prev_button = gr.Button("⬅️ 上一条")
                next_button = gr.Button("下一条 ➡️")
            index_input = gr.Number(label="跳转到索引 (从0开始)", value=0, precision=0)
            go_button = gr.Button("跳转", variant="primary")
        
        with gr.Column(min_width=300):
            gr.Markdown("#### 状态")
            status_text = gr.Textbox(label="当前位置", interactive=False)
            rating_status_text = gr.Textbox(label="评分状态", interactive=False)
        
        with gr.Column(min_width=300):
            # 【核心修改】分析框现在直接显示，不再需要按钮
            gr.Markdown("#### 实时分析结果")
            analysis_output = gr.Textbox(label="评分分布", lines=4, interactive=False)

    # --- 5. 事件处理逻辑 ---
    
    # 【核心修改】评分选项改变时，调用新函数更新状态和分析结果
    score_radio.change(
        update_score_and_analysis, 
        inputs=[current_index, score_radio, total_samples], 
        outputs=[rating_status_text, analysis_output]
    )

    def go_to_and_update(index, total):
        new_index = int(index) if total > 0 and 0 <= int(index) < total else current_index.value
        html_a, html_b, choice, status, rating_status = get_sample(new_index)
        return new_index, html_a, html_b, choice, status, rating_status

    def prev_sample(index, total): return go_to_and_update(max(0, int(index) - 1), total)
    def next_sample(index, total): return go_to_and_update(min(total - 1, int(index) + 1), total)

    outputs_for_nav = [current_index, prediction_a_html, prediction_b_html, score_radio, status_text, rating_status_text]
    
    prev_button.click(prev_sample, inputs=[current_index, total_samples], outputs=outputs_for_nav)
    next_button.click(next_sample, inputs=[current_index, total_samples], outputs=outputs_for_nav)
    go_button.click(go_to_and_update, inputs=[index_input, total_samples], outputs=outputs_for_nav)
    index_input.submit(go_to_and_update, inputs=[index_input, total_samples], outputs=outputs_for_nav)

    def on_load():
        num_samples = load_data()
        load_scores()
        html_a, html_b, choice, status, rating_status = get_sample(0)
        # 【核心修改】加载时也计算一次分析结果
        stats = get_score_stats(num_samples)
        analysis_text = generate_analysis_text(stats)
        return num_samples, 0, html_a, html_b, choice, status, rating_status, analysis_text
    
    # 【核心修改】demo.load的输出增加了 analysis_output
    demo.load(on_load, outputs=[total_samples, current_index, prediction_a_html, prediction_b_html, score_radio, status_text, rating_status_text, analysis_output])

# --- 6. 启动应用 ---
if __name__ == "__main__":
    demo.launch()
    
    