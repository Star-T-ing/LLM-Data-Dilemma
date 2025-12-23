# Please install OpenAI SDK first: `pip3 install openai`
import os
import random
from tqdm import trange
from openai import OpenAI

base_setting = {
    'qwen': {
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        'model': 'qwen-long'
    },
    'deepseek': {
        'url': 'https://api.deepseek.com',
        'model': 'deepseek-chat'
    },
    'doubao': {
        'url': 'https://ark.cn-beijing.volces.com/api/v3',
        'model': 'doubao-seed-1-6-flash-250828'
    }
}

STYLES = [
    "正式新闻报道风格",
    "简洁客观的事实陈述风格",
    "偏口语但保持新闻客观性的风格",
    "书面化、逻辑严谨的新闻风格",
    "通俗易懂、面向大众读者的新闻风格"
]

def build_rewrite_prompt(title, content):
    style = random.choice(STYLES)
    length = len(content)

    min_len = int(length * 0.9)
    max_len = int(length * 1.1)

    prompt = f"""
    你是一名专业的中文新闻编辑。

    请在【不改变事实、不引入新信息、不进行总结或扩展】的前提下，
    对给定的新闻正文进行改写，使其与原文语义一致，但表达方式不同。

    【写作要求】
    1. 写作风格：{style}
    2. 文本类型：新闻正文
    3. 输出语言：中文
    4. 保留新闻的关键信息、时间、人物和事件关系
    5. 不要添加评论性语言或主观判断
    6. 避免使用与原文高度重复的句式

    【长度要求】
    - 输出长度控制在 {min_len} 到 {max_len} 字之间

    【新闻标题】
    {title}

    【原始新闻正文】
    {content}

    【改写后的新闻正文】
    """
    return prompt

def regenerate(title: str, content: str, api: str):
    api_upper = f"{api.upper()}_APIKEY"
    api_key = os.getenv(api_upper)
    client = OpenAI(
        # api_key = 'sk-4d4a7ad014a74ed4a9ce15628bfaa347',
        api_key = api_key,
        base_url=base_setting[api]['url']
    )
    
    prompt = build_rewrite_prompt(title, content)
    
    response = client.chat.completions.create(
        model=base_setting[api]['model'],
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content
    
    return result
