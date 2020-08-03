# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import json
import sys

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm
from collections import OrderedDict
from models.modeling_bert import Config, QuestionAnswering
from utils.korquad_utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions, InputFeatures, SquadExample)
from utils.tokenization import BertTokenizer
from debug.evaluate_korquad import evaluate as korquad_eval
# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)


logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def write_predict(args, model, eval_examples, eval_features, file_num, parse_num):
    """ Eval """
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    logger.info("***** Evaluating *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    all_results = []
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    logger.info("Start evaluating!")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}_{}.json".format(file_num, parse_num))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}.json".format(file_num, parse_num))
    write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      False, output_prediction_file, output_nbest_file,
                      None, False, False, 0.0)


def save_examples(args, examples, num1, num2):
    output_examples = "eval_example_{0}_{1}.pkl".format(num1, num2)
    output_exam_file = os.path.join(args.example_dir, output_examples)
    with open(output_exam_file, 'wb') as output:
        pickle.dump(examples,output,pickle.HIGHEST_PROTOCOL)
    logger.info("extracting end, %s saved", output_examples)

def save_features(args, features, num1, num2):
    output_features = "eval_feature_{0}_{1}.pkl".format(num1, num2)
    output_feat_file = os.path.join(args.example_dir, output_features)
    with open(output_feat_file, 'wb') as output:
        pickle.dump(features, output, pickle.HIGHEST_PROTOCOL)
        
    logger.info("extracting end, %s saved", output_features)

def load_and_cache_examples(args, tokenizer):
    # Load data features from cache or dataset file
    for a in range(args.extract_file_start,5):
        a='0'+str(a)
        filename = args.predict_file+a+'.json'
        with open(filename, encoding="utf-8") as data_file:
            datas = json.load(data_file, object_pairs_hook=OrderedDict)
        i=0
        if args.extract_start>0:
            i=args.extract_start
            args.extract_start = 0
        while (i+1)*args.parse_size<= len(datas["data"]):
            data = datas["data"][i*args.parse_size:(i+1)*args.parse_size]
            examples = read_squad_examples(data= data,
                                        is_training=False,
                                        version_2_with_negative=False)
            features = convert_examples_to_features(args,
                                                    examples=examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=args.max_seq_length,
                                                    doc_stride=args.doc_stride,
                                                    max_query_length=args.max_query_length,
                                                    is_training=False,
                                                    file_num=a)
            save_examples(args, examples, a, str(i))
            save_features(args, features, a, str(i))
            i+=1


def main():
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument("--output_dir", default='debug', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='output/korquad_3.bin', type=str,
                        help="fine-tuned model checkpoint")
    parser.add_argument("--config_file", default='data/large_config.json', type=str,
                        help="model configuration file")
    parser.add_argument("--vocab_file", default='data/large_v1_32k_vocab.txt', type=str,
                        help="tokenizer vocab file")

    parser.add_argument("--predict_file", default='dev/korquad2.1_dev_', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=64, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--examples_data_start", default = 0, type= int)
    parser.add_argument("--extract", default = False, type = bool,
                        help="Extract examples and features from dev file")
    parser.add_argument("--load_and_predict", default = False, type = bool,
                        help = "load examples and feature files and evaluate")
    parser.add_argument("--example_dir", default='dev/examples/', type=str,
                        help="examples file directory path")
    
    parser.add_argument("--load_start", default=0, type= int,
                        help="load file start")
    parser.add_argument("--load_start_parse", default=0, type= int,
                        help="if load file parsed, parse start")
    parser.add_argument("--extract_start", default=0, type= int,
                        help="extract file start")
    parser.add_argument("--extract_file_start", default=0, type = int)
    parser.add_argument("--parse_size", default=250, type = int,
                        help="for Out Of Memory problem parse data files")
    parser.add_argument("--evaluate", default=False, type=bool)
    parser.add_argument("--save_time", default=7200, type=int)

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions."
                             "json output file.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.info("device: %s, n_gpu: %s, 16-bits training: %s", device, args.n_gpu, args.fp16)

    # Set seed
    set_seed(args)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_basic_tokenize=True, max_len=args.max_seq_length)
    config = Config.from_json_file(args.config_file)
    model = QuestionAnswering(config)
    model.load_state_dict(torch.load(args.checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    if args.fp16:
        model.half()
    model.to(args.device)
    logger.info("Evaluation parameters %s", args)

    # Evaluate
    if args.extract is True:
        load_and_cache_examples(args, tokenizer)
    examples = []
    features = []
    #split data and split predict
    if args.load_and_predict is True:
        for a in range(args.load_start, 5):
            a = '0'+str(a)
            for i in range(args.load_start_parse, 40):
                load_exam = args.example_dir+"eval_{}_{}_{}.pkl".format('example',a,str(i))
                load_feat = args.example_dir+"eval_{}_{}_{}.pkl".format('feature',a,str(i))
                if not os.path.exists(load_exam):
                    continue
                logger.info("load examples and features in file %s", a+'_'+str(i))
                with open(load_exam,'rb') as input:
                    example = pickle.load(input)
                    for e in example:
                        examples.append(e)
                with open(load_feat,'rb') as input:
                    feature = pickle.load(input)
                    for f in feature:
                        features.append(f)
                write_predict(args, model, examples, features, a, i)
                del features[:]
                num=1
                load_feat_extra = args.example_dir+"eval_{}_{}_{}_{}.pkl".format('feature',a,str(i), str(num))
                while os.path.exists(load_feat_extra):
                    logger.info("load_extra_feature_file_{}".format(str(num)))
                    with open(load_feat_extra,'rb') as input:
                        feature = pickle.load(input)
                        for f in feature:
                            features.append(f)
                            
                    write_predict(args, model, examples, features, a, str(i)+'_'+str(num))
                    del features[:]
                    num+=1
                    load_feat_extra = args.example_dir+"eval_{}_{}_{}_{}.pkl".format('feature',a,str(i), str(num))
                del examples[:]
        #after write predictions merge prediction files in "predictions.json"
    if args.evaluate is True:
        with open(os.path.join(args.output_dir, "predictions.json")) as prediction_file:
            predictions = json.load(prediction_file)
        logger.info(json.dumps(korquad_eval(args, predictions)))


if __name__ == "__main__":
    main()
