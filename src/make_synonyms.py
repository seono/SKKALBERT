from utils.models.word_eval import WordEmbeddingEvaluator
import json
import collections
import pickle
import khaiii
import os
import re
import khaiii
from tqdm import tqdm, trange
api = khaiii.KhaiiiApi()
api.open()

REMOVE_ENTER = re.compile("\\n")

SAVE_TAG = ["NNG", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN"]


def WordEmbeddingModel_init(model_name):

    global embedd_model
    if model_name == "fasttext":
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/fasttext/fasttext.vec",
			vecs_bin_fname = "notebooks/embedding/data/word-embeddings/fasttext/fasttext.bin",
			method = "fasttext", dim=100, tokenizer_name="khaiii")
    elif model_name == "glove":
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/glove/glove.txt",
            method="glove", dim=100, tokenizer_name="khaiii")
    elif model_name == "swivel":
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/swivel/row_embedding.tsv",
			method="swivel", dim=100, tokenizer_name="khaiii")
    else:
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/word2vec/word2vec",
            method="word2vec", dim=100, tokenizer_name="khaiii")

def get_words():
	with open("data/korquad/KorQuAD_v1.0_train.json","r", encoding='utf-8') as p:
		data = json.load(p)["data"]
	question_words = set()
	for entry in tqdm(data, total=len(data), desc="reading...", leave=True, position=0):
		for para in entry["paragraphs"]:
			for qa in para["qas"]:
				q_text=qa["question"]
				q_text = re.sub(REMOVE_ENTER, ' ',q_text)
				for word in api.analyze(q_text):
					for morph in word.morphs:
						if morph.tag in SAVE_TAG:
							question_words.add(morph.lex)
							
	with open("data/korquad/KorQuAD_v1.0_dev.json","r", encoding='utf-8') as p:
		data = json.load(p)["data"]
	
	for entry in tqdm(data, total=len(data), desc="reading...", leave=True, position=0):
		for para in entry["paragraphs"]:
			for qa in para["qas"]:
				q_text=qa["question"]
				q_text = re.sub(REMOVE_ENTER, ' ',q_text)
				for word in api.analyze(q_text):
					for morph in word.morphs:
						if morph.tag in SAVE_TAG:
							question_words.add(morph.lex)

	return list(question_words)

if __name__ == "__main__":
	WordEmbeddingModel_init("word2vec")
	question_words = get_words()
	with open("question_word_set.pkl", "wb") as p:
		pickle.dump(question_words,p)
