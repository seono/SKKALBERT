from __future__ import print_function
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
import logging

'''KorQuAD v1.0에 대한 공식 평가 스크립트 '''
'''본 스크립트는 SQuAD v1.1 평가 스크립트 https://rajpurkar.github.io/SQuAD-explorer/ 를 바탕으로 작성됨.'''

logger = logging.getLogger(__name__)

def normalize_answer(s):    
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)      
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
   
    #F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)
        
    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)   
        
    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    return metric_fn(prediction, ground_truth)


def evaluate(args, predictions):
    f1 = exact_match = total = 0
    expected_version = 'KorQuAD_2.0'
    predict_file = args.predict_file
    for a in range(0,5):
        a='0'+str(a)
        filename = predict_file+a+'.json'
        with open(filename, encoding="utf-8") as data_file:
            datas = json.load(data_file, object_pairs_hook=OrderedDict)
            read_version = "_".join(datas['version'].split("_")[:-1])
            if (read_version != expected_version):
                logger.info('Evaluation expects ' + expected_version +
                            ', but got dataset with ' + read_version)
        i=0
        while (i+1)*args.parse_size<= len(datas["data"]):
            logger.info("evaluate data {}_{}".format(a,str(i)))
            dataset = datas["data"][i*args.parse_size:(i+1)*args.parse_size]
            for paragraph in dataset:
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truth = qa['answer']['text']
                    prediction = predictions[qa['id']]
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truth)
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truth)
            i+=1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = 'KorQuAD_2.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for KorQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            print('Evaluation expects ' + expected_version +
                  ', but got dataset with ' + read_version,
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(args, predictions)))
