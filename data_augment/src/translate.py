# -*- coding: utf-8 -*-

import os
import json
import types
import time
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tmt.v20180321 import tmt_client, models

def translate(q, src_lang="zh", tgt_lang="en"):
    try:
        SecretId = os.getenv('SECRET_ID')
        SecretKey = os.getenv('SECRET_KEY')
        cred = credential.Credential(SecretId, SecretKey)
        # 使用临时密钥示例
        # cred = credential.Credential("SecretId", "SecretKey", "Token")
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tmt.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = tmt_client.TmtClient(cred, "ap-beijing", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.TextTranslateRequest()
        params = {
            "SourceText": q,
            "Source": src_lang,
            "Target": tgt_lang,
            "ProjectId": 0
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个TextTranslateResponse的实例，与请求对象对应
        resp = client.TextTranslate(req)
        # 输出json格式的字符串回包
        return resp.TargetText

    except TencentCloudSDKException as err:
        print(err)
        

def back_translate(sample, src_lang="zh", tgt_lang="en"):
    """
    q: 输入的文本
    src_lang: 源语言
    tgt_lang: 目标语言
    """
    limits = 700
    if len(sample) > limits:
        return None
    en = translate(sample, src_lang, tgt_lang)
    time.sleep(1)
    target = translate(en, tgt_lang, src_lang)
    time.sleep(1)
    return target


if __name__ == '__main__':
    sample = "范甘迪最近在接受采访时就谈到了两人目前的关系，大范说：“我很久没和帕特聊天了，我不像其他人那样把这个很当一回事。"
    print(back_translate(sample))
    # print(translate(sample, 'zh', 'en'))