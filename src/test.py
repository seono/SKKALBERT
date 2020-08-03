from utils.bs4 import get_wiki_context
import argparse
import json
import os
import sys
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--predict_file", default='dev/korquad2.1_dev_', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--train_file_name", default='train/korquad2.1_train_', type=str,
                        help="KorQuAD json directory for training. E.g., train")
    args = parser.parse_args()
    answer_tag_list = set()
    test_case = 3
    for i in range(1):
        i = '0'+str(i)
        file_path = args.predict_file+i+'.json'
        with open(file_path,encoding='utf-8') as data_file:
            datas = json.load(data_file, object_pairs_hook=OrderedDict)
            text = datas["data"][0]['context']
        temp = get_wiki_context(text)
    print(temp[:10])
    for t in temp[:10]:
        print(t)
if __name__ == "__main__":
    main()