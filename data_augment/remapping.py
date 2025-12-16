import os
import os
import shutil
import json
import logging
import argparse
from tqdm import trange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    if os.path.isdir(args.orin_dir):
        orin_list = os.listdir(args.orin_dir)
        output_list = os.listdir(args.output_dir)
        output_count = len(output_list)
        
        with open(args.mapping_path, 'a+', encoding='utf-8') as mp:
            for i, filename in enumerate(orin_list):
                orin_path = os.path.join(args.orin_dir, filename)
                # cleaned_sample_31.txt
                prefix_list = filename.split('_')
                prefix_list[2] = f'{output_count + i + 1}' + '.txt'
                output_name = '_'.join(prefix_list)
                output_path = os.path.join(args.output_dir, output_name)
                shutil.copy(orin_path, output_path)
                new_dict = {
                    "cur_name": output_name,
                    "orin_name": filename,
                    "method": args.method
                }
                data = json.dumps(new_dict)
                mp.write(data)
                mp.write('\n')
        
    else:
        logging.warning('Please enter the existing oringinal dataset dir')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orin_dir", default="/home/ruansikai/Limerence/assignments/LLM/translate")
    parser.add_argument("--output_dir", default="/home/ruansikai/Limerence/assignments/LLM/augmented_data")
    parser.add_argument("--method", choices=['translate', 'replace', 'regenerate', 'structure'])
    parser.add_argument("--mapping_path", default="/home/ruansikai/Limerence/assignments/LLM/mapping.jsonl")
    
    args = parser.parse_args()
    
    main(args)