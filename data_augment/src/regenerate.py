# Please install OpenAI SDK first: `pip3 install openai`
import os
from tqdm import trange
from openai import OpenAI

def regenerate(sample):
    client = OpenAI(
        api_key = os.getenv('API_KEY'),
        base_url="https://api.deepseek.com"
    )
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "重写输入的新闻，要求字数一致"},
            {"role": "user", "content": sample}
        ]
    )
    result = response.choices[0].message.content
    
    return result
