import json
import logging
import math
import collections
import os
from tqdm import tqdm, trange
from io import open
from .eda import eda_one_op, eda
from time import gmtime, strftime

logger = logging.getLogger(__name__)


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_squad_train_examples(data_dir, filename=None, args=None):
    
    if data_dir is None:
        data_dir = ""
        
    if args.train_file is None:
        raise ValueError("Train file must be specified")
    
    with open(os.path.join(data_dir, args.train_file if filename is None else filename), "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    
    # Original sentences
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []
                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False
                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]
                else:
                    answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
        
                # Add augmented sentences  
                if args.op is not None:
                    aug_examples = augment_question(example, args, start_position_character)
                    for aug_example in aug_examples:
                        examples.append(aug_example)
    return examples


def augment_question(example, args, start_position_character):
    # print(example.question_text)
    if args.eda_all_op:
        aug_sentences = eda(example.question_text, alpha_sr=args.alpha, alpha_ri=args.alpha, alpha_rs=args.alpha, p_rd=args.alpha, num_aug=args.num_aug)
    else:
        aug_sentences = eda_one_op(example.question_text, args.op, alpha=args.alpha, num_aug=args.num_aug)
    # print(aug_sentences)
    aug_examples = []
    # print(example.start_position)
    for aug_sentence in aug_sentences:
        new_example = SquadExample(
                        qas_id=example.qas_id,
                        question_text=aug_sentence,
                        context_text=example.context_text,
                        answer_text=example.answer_text,
                        start_position_character=start_position_character,
                        answers=example.answers,
                        title=example.title,
                        is_impossible=example.is_impossible)
        aug_examples.append(new_example)
    return aug_examples

def get_now_str():
    return str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))