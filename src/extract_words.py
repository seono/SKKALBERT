import json
import collections
import re
import khaiii
from tqdm import tqdm, trange
api = khaiii.KhaiiiApi()
api.open()

REMOVE_ENTER = re.compile("\\n")

SAVE_TAG = ["NNG", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN"]

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
question_words = list(question_words)
print((question_words[:1000]))
print(len(question_words))