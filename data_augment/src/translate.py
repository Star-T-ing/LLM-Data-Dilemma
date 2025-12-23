# -*- coding: utf-8 -*-

import os
import json
import types
import time
import http
import random
import json
import hashlib
import urllib
from sentence_transformers import SentenceTransformer, util

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tmt.v20180321 import tmt_client, models

from alibabacloud_alimt20181012.client import Client as alimt20181012Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_alimt20181012 import models as alimt_20181012_models
from alibabacloud_tea_util import models as util_models

threshold = 0.8

def aliyun_translate(q, src_lang="zh", tgt_lang="en"):
    ACCESS_KEY_ID = 'Access_key_id'
    ACCESS_KEY_SECRET = 'Access_key_secret'

    config = open_api_models.Config(
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET
    )
    config.endpoint = f'mt.cn-hangzhou.aliyuncs.com'
    client = alimt20181012Client(config)

    translate_general_request = alimt_20181012_models.TranslateGeneralRequest(
        format_type = 'text',
        source_language = src_lang,
        target_language = tgt_lang,
        source_text = q,
        scene = 'general'
    )
    runtime = util_models.RuntimeOptions()
    resp = client.translate_general_with_options(translate_general_request, runtime)
    return resp.body.data.__dict__['translated']


def tencent_translate(q, src_lang="zh", tgt_lang="en"):
    try:
        SecretId = 'SecretId'
        SecretKey = 'SecretKey'
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
        
def baidu_translate(q, src_lang="zh", tgt_lang="en"):
    appid = os.getenv('BAIDU_APPID')
    secretKey = os.getenv('SECRET_KEY')

    httpClient = None
    myurl = '/api/trans/vip/translate'

    salt = random.randint(0, 4000)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = '/api/trans/vip/translate' + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + src_lang + '&to=' + tgt_lang + '&salt=' + str(salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        return result['trans_result'][0]['dst']

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()
    

def cosine_simalarity(source, target):
    model = SentenceTransformer('shibing624/text2vec-base-chinese')

    emb1 = model.encode(source, convert_to_tensor=True)
    emb2 = model.encode(target, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    similarity = round(similarity, 4)  
    
    if similarity < threshold:
        return None
    else:
        return target
    

def back_translate(sample, api = 'tencent', src_lang="zh", tgt_lang="en"):
    """
    q: 输入的文本
    src_lang: 源语言
    tgt_lang: 目标语言
    """
    if api == 'tencent':
        en = tencent_translate(sample, src_lang, tgt_lang)
        time.sleep(1)
        target = tencent_translate(en, tgt_lang, src_lang)
        time.sleep(1)
    elif api == 'baidu':
        en = baidu_translate(sample, src_lang, tgt_lang)
        time.sleep(1)
        target = baidu_translate(en, tgt_lang, src_lang)
        time.sleep(1)
    elif api == 'aliyun':
        en = aliyun_translate(sample, src_lang, tgt_lang)
        time.sleep(1)
        target = aliyun_translate(en, tgt_lang, src_lang)
        time.sleep(1)
    return cosine_simalarity(sample, target)


if __name__ == '__main__':
    sample = "范甘迪最近在接受采访时就谈到了两人目前的关系，大范说：“我很久没和帕特聊天了，我不像其他人那样把这个很当一回事。"
    print(back_translate(sample))
