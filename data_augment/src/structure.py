import os
from openai import OpenAI

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
api_key = os.getenv('ARK_API_KEY')

def structure(input_text, model):
    news_length = len(input_text)
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=api_key,
    )

    structure = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "结构化输入的文本"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": input_text
                    }
                ]
            }
        ],
    )

    structure_text = structure.choices[0].message.content
    rewrite = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"根据输入的结构化文本，写一篇新闻，字数为{news_length}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": structure_text
                    },
                ],
            }
        ]
    )

    return rewrite.choices[0].message.content

if __name__ == "__main__":
    input_text = "“我的淘宝旺旺经常会收到网店的宣传消息”“我卖纸箱卖了1年才到皇冠，他们刷1个月不到，就皇冠了，心理严重失衡啊”……由于部分用户涉及重度虚假交易以及滥发宣传信息，记者昨日获悉，淘宝网目前开始加大力度查处不诚信卖家，目前，查封的淘宝网店数已近3000家。 据淘宝方面透露，该网站的第二代安全稽查监控系统已经正式上线，将实时对虚假交易行为进行更有效的监控，最近再次查封了691个炒作账户，截至目前，已经有2989家店被查封，查封的原因为这批网店涉及重度虚假交易。 据了解，除了重度虚假交易被查封外，网上发送垃圾信息也将成为打击重点。淘宝公告显示，由于部分用户长期、大量地发送垃圾消息，对阿里旺旺用户造成了严重的骚扰，淘宝近日永久封杀了川味坊等4家网店。"
    print(structure(input_text))