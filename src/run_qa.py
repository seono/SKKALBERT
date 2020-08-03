from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import json

import numpy as np
import torch


from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool, cpu_count

from models.modeling_bert import QuestionAnswering, Config
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.korquad_utils import KorquadV2Processor, convert_examples_to_features, write_predictions, korquad_convert_examples_to_features_init, korquad_convert_ex_to_ft
from debug.evaluate_korquad import evaluate as korquad_eval

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_predict(args, model, eval_examples, eval_features):
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
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      False, output_prediction_file, output_nbest_file,
                      None, False, False, 0.0)

def train(args, train_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size // args.gradient_accumulation_steps
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=t_total*args.warmup_proportion, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.fp16_opt_level == "O2":
            keep_batchnorm_fp32 = False
        else:
            keep_batchnorm_fp32 = True
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level, keep_batchnorm_fp32=keep_batchnorm_fp32
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        for step, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            batch = tuple(t.to(args.device) for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            outputs = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            loss = outputs  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule\
                
                optimizer.step()
                optimizer.zero_grad()
    logger.info("Training End!!!")


def load_and_cache_examples(args, tokenizer):
    # Load data features from cache or dataset file
    # one input file, change to diretory
    train_files = [temp_file for temp_file in os.listdir(args.train_dir) if '.json' in temp_file]
    processor = KorquadV2Processor(args.threads)
    examples = []
    if args.evaluate:
        ## Find json file name
        predict_files = [temp_file for temp_file in os.listdir(args.predict_dir) if '.json' in temp_file]

        ## Load json files
        for predict_file in predict_files:
            logger.info("Reading examples from dataset file at %s", predict_file)
            temp_examples = processor.get_dev_examples(args.predict_dir, filename=predict_file)
            if temp_examples is not None and len(temp_examples)>0:
                examples.extend(temp_examples)
    else:
        train_files = [temp_file for temp_file in os.listdir(args.train_dir) if '.json' in temp_file]

        ## Load json files
        for train_file in train_files:
            logger.info("Reading examples from dataset file at %s", train_file)
            temp_examples = processor.get_train_examples(args.train_dir, filename=train_file)
            if temp_examples is not None and len(temp_examples) > 0:
                examples.extend(temp_examples)


    is_training = True
    if args.evaluate is True:
        is_training = False
    
    
    return korquad_convert_ex_to_ft(examples, 
                            max_seq_length = args.max_seq_length,
                            doc_stride = args.doc_stride,
                            max_query_length = args.max_query_length,
                            is_training = is_training,
                            tokenizer=tokenizer)
    

def save_examples(args, dataset, num1, num2):
    output_examples = "korquad_example_{0}_{1}.bin".format(num1, num2)
    logger.info("extracting end, %s saved", output_examples)
    if not os.exists(args.example_dir):
        os.makedirs(args.example_dir)
    output_exam_file = os.path.join(args.example_dir, output_examples)
    torch.save(dataset, output_exam_file)

def main():
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='pretrain_ckpt/large_v1_model.bin', type=str,
                        help="pre-trained model checkpoint")
    parser.add_argument("--config_file", default='data/large_config.json', type=str,
                        help="model configuration file")
    parser.add_argument("--vocab_file", default='data/large_v1_32k_vocab.txt', type=str,
                        help="tokenizer vocab file")

    parser.add_argument("--train_dir", default='train/', type=str,
                        help="KorQuAD json directory for training. E.g., train")

    parser.add_argument("--evaluate", default=False, type=bool)

    parser.add_argument("--threads", default=1, type=int)

                        
    parser.add_argument("--test", default = False, type=bool)
    parser.add_argument("--examples_data_start", default = 0, type= int)

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=300, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--parse_size", default=4, type=int,
                        help="for downsize memory parse dataset")
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', default='O2', type=str,
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")


    parser.add_argument('--parse_data', type=bool, default=True,
                        help="Resolve Out of Memory")
    args = parser.parse_args()

    
    parse_size = args.parse_size

    tokenizer = BertTokenizer(args.vocab_file, max_len=args.max_seq_length, do_basic_tokenize=True)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)


    # Prepare model
    config = Config.from_json_file(args.config_file)
    model = QuestionAnswering(config)
    # checkpoint에서... 
    model.load_state_dict(torch.load(args.checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)

    logger.info("Training hyper-parameters %s", args)

    
    logger.info("Parse and extract example")
    
    
    if args.evaluate:
        examples, features = load_and_cache_examples(args, tokenizer)
        write_predict(args, model,examples, features)
        with open(os.path.join(args.output_dir, "predictions.json")) as prediction_file:
            predictions = json.load(prediction_file)
        logger.info(json.dumps(korquad_eval(args, predictions)))
    else:
        data_set = load_and_cache_examples(args, tokenizer)
        train(args, data_set, model)
        if args.local_rank in [-1, 0]:
            model_checkpoint = "korquad_{0}_{1}_{2}_{3}.bin".format(args.learning_rate,
                                                                args.train_batch_size,
                                                                e,
                                                                e_)
            logger.info(model_checkpoint)
            output_model_file = os.path.join(args.output_dir, model_checkpoint)
            if args.n_gpu > 1 or args.local_rank != -1:
                logger.info("** ** * Saving file * ** ** (module)")
                torch.save(model.module.state_dict(), output_model_file)
            else:
                logger.info("** ** * Saving file * ** **")
                torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()
