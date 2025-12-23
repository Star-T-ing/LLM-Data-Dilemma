import os
from openai import OpenAI

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
# api_key = os.getenv('ARK_API_KEY')
api_key = os.getenv('DOUBAO_APIKEY')

def structure(title: str, body: str, model: str):
    news_length = len(body)
    prompt_structure = f'''
    你是一名信息抽取系统，负责从中文新闻文本中提取结构化事实信息。

    请严格依据原文内容进行抽取，不要进行改写、总结、推测或补充，
    所有信息必须能够在原文中找到明确依据。

    【抽取要求】
    1. 仅输出结构化结果，不要输出任何解释性文字
    2. 若某一字段在原文中不存在，请置为空字符串
    3. 保持字段语义准确，避免冗余描述

    【输出格式（JSON）】
    {
        "event_type": "",
        "time": "",
        "location": "",
        "subjects": [],
        "objects": [],
        "key_actions": "",
        "cause": "",
        "result": "",
        "additional_details": ""
    }

    【新闻正文】
    {body}
    '''
    
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    structure = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt_structure
                    }
                ]
            }
        ],
        response_format={"type": "json_object"}
    )

    structure_format = structure.choices[0].message.content
    prompt_regenerate = f'''
        你是一名专业的中文新闻编辑。
        请根据给定的结构化新闻信息，生成一篇完整、通顺的新闻正文。
        生成内容必须严格基于提供的信息，不得引入任何原始结构中未包含的新事实。

        【生成要求】
        1. 文本类型：新闻正文
        2. 输出语言：中文
        3. 保持新闻客观性，不加入评论或主观判断
        4. 不逐条照搬结构字段，应进行自然语言组织
        5. 不得编造时间、地点、人物或事件细节

        【长度要求】
        生成文本长度与常规新闻报道相当（约 {news_length * 0.8}–{news_length * 1.2} 字）

        【结构化新闻信息（JSON）】
        {structure_format}

        【生成的新闻正文】
    '''
    rewrite = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_regenerate
                    },
                ],
            }
        ]
    )

    return rewrite.choices[0].message.content

if __name__ == "__main__":
    pass
