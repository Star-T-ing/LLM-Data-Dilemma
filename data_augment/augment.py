import os
import random
import logging
import argparse
from tqdm import trange
from src.extract_main import extract
from src.translate import back_translate
from src.replace import replace
from src.regenerate import regenerate
from src.structure import structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

support_language = ['en', 'ko', 'de']

def store_single(output_dir, sample, output):
    with open(os.path.join(output_dir, sample['path']), 'w', encoding='utf-8') as f:
        if output == '' or output is None:
            output = sample['body']
        data = sample['title'] + ' ' + output.replace('\n', '')
        f.write(data)

def store_list(output_dir, sample_list, output_list):
    for i in range(len(output_list)):
        store_single(output_dir, sample_list[i], output_list[i])

def main(args):
    if os.path.isdir(args.orin_dir):
        samples = os.listdir(args.orin_dir)
        done_list = os.listdir(args.output_dir)
        sample_list, input_list = [], []
        for sample in samples:
            with open(os.path.join(args.orin_dir, sample), 'r', encoding='utf-8') as f:
                sample_data = f.read()
                title, body = extract(sample_data)
                if title is None or body is None:
                    continue
                sample_list.append({'title': title, 'body': body, 'path': sample})
                input_list.append(body)
        
        total_length = len(sample_list)
        n_share = args.n_share
        n_size = total_length // n_share
        idx = args.idx
        
        sample_list = sample_list[(idx * n_size):((idx + 1) * n_size)]
        input_list = input_list[(idx * n_size):((idx + 1) * n_size)]
        # input_list = input_list[:10]
        # sample_list = sample_list[:10]
        
        result_list = []
        
        if args.method == 'translate':
            for i in trange(len(sample_list)):
                sample = sample_list[i]
                if sample['path'] in done_list:
                    continue
                tgt_lang = random.choice(support_language)
                output = back_translate(sample['body'], api=args.transapi, src_lang='zh', tgt_lang=tgt_lang)
                if output is None:
                    continue
                store_single(args.output_dir, sample, output)
        elif args.method == 'replace':
            output_list = replace(input_list)
            store_list(args.output_dir, sample_list, output_list)
        elif args.method == 'regenerate':
            for i in trange(len(sample_list)):
                sample = sample_list[i]
                if sample['path'] in done_list:
                    continue
                try:
                    output = regenerate(sample['title'], sample['body'], 'deepseek')
                    store_single(args.output_dir, sample, output)
                except Exception as err:
                    print(err)
                    continue
        elif args.method == 'structure':
            model_list = ['deepseek-v3-2-251201', 'deepseek-v3-250324', 
                          'deepseek-r1-250528', 'doubao-seed-1-6-251015', 'doubao-seed-1-6-flash-250828']
            for i in trange(len(sample_list)):
                model = random.choice(model_list)
                sample = sample_list[i]
                if sample['path'] in done_list:
                    continue
                try:
                    output = structure(sample['title'], sample['body'], model)
                    store_single(args.output_dir, sample, output)
                except Exception as err:
                    print(err)
                    continue
    else:
        logging.warning('Please enter the existing oringinal dataset dir')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orin_dir", default="/home/ruansikai/Limerence/assignments/LLM/datasets/cleaned_data")
    parser.add_argument("--output_dir", default="/home/ruansikai/Limerence/assignments/LLM/translate")
    parser.add_argument("--method", choices=['translate', 'replace', 'regenerate', 'structure', 'all'])
    parser.add_argument("--trans_api", choices=['aliyun', 'tencent', 'baidu'])
    parser.add_argument("--n_share", type=int, default=8)
    parser.add_argument("--idx", type=int, default=4)
    
    args = parser.parse_args()
    
    main(args)
