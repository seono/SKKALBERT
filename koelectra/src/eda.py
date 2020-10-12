import random
from random import shuffle
random.seed(1)
from .models.word_eval import WordEmbeddingEvaluator
import os
import re
import khaiii
import pickle
api = khaiii.KhaiiiApi()
api.open()

REMOVE_ENTER = re.compile("\\n")

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
            vecs_txt_fname = "embeddings/word2vec",
            method="word2vec", dim=100, tokenizer_name="khaiii")

def synonym_dict_init(dict_file_name):
    global synonym_dict
    if os.path.exists(dict_file_name):
        print("load file {}".format(dict_file_name))
        with open(dict_file_name,"rb") as p:
            synonym_dict = pickle.load(p)
    
def synonym_replacement(sentence, n, min_score):
    SAVE_TAG = ["NNG", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN"]
    sentence = re.sub(REMOVE_ENTER, ' ', sentence)
    target_idx = []
    words = []
    i = 0
    num_replaced=0
    for word in api.analyze(sentence):
        for m in word.morphs:
            words.append(m.lex)
            if m.tag in SAVE_TAG and len(m.lex)>1:
                target_idx.append(i)
            i+=1
    random.shuffle(target_idx)
    for random_idx in target_idx:
        synonyms = get_synonyms(words[random_idx], min_score)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            words[random_idx] = synonym
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    return words


def get_synonyms(word, min_score=0.85):
    synonyms = set()
    if(len(synonym_dict)>0):
        if word in synonym_dict:
            for w, score in synonym_dict[word][:10]:
                if(score<min_score):
                    break
                synonyms.add(w)
        return list(synonyms)
    else:
        for w, score in embedd_model.most_similar(word,topn=10):
            if(score<min_score):
                break
            synonyms.add(w)
        return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, n):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = [word for word in words]
    i = 0
    x = 0
    while i<len(new_words):
        p = random.uniform(0,1)
        if p>0.5:
            new_words.pop(i)
            x+=1
            if x==n:
                break
        else:
            i+=1


	#if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, min_score):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, min_score)
    return new_words

def add_word(new_words, min_score):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word,min_score)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

#######################################################################################
# In korean, it looks similar in meaning inserting synonym words to it's vicinity.
# So, i make ver.2 of random insert eda
#######################################################################################

def random_insertion_ver2(words, n, min_score):
    new_words = words.copy()
    for _ in range(n):
        add_word_ver2(new_words, min_score)

    return new_words

def add_word_ver2(new_words, min_score):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        rand_index = random.randint(0,len(new_words)-1)
        random_word = new_words[rand_index]
        synonyms = get_synonyms(random_word,min_score)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    rand_ver2 = [rand_index,rand_index+1]
    #input synonyms front or back
    random_index = random.choice(rand_ver2)
    new_words.insert(random_index, random_synonym)

########################################################################
# main data augmentation function
########################################################################
def test_eda(sentence):
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = 1
    n = max(1, int(0.1*num_words))
    a_words = synonym_replacement(sentence, n, 0.85)
    augmented_sentences.append(' '.join(a_words))
    a_words = random_insertion_ver2(words, n, 0.85)
    augmented_sentences.append(' '.join(a_words))
    a_words = random_swap(words, n)
    augmented_sentences.append(' '.join(a_words))
    a_words = random_deletion(words, n)
    augmented_sentences.append(' '.join(a_words))

    return augmented_sentences


def eda(sentence, eda_type, alpha=0.1, num_aug=1, min_score=0.7):
    #remove ?
    sentence= sentence[:-1]
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = num_aug
    n = max(1, int(alpha*num_words))

    #sr
    if eda_type == "sr":
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(sentence+"?", n, min_score)
            augmented_sentences.append(' '.join(a_words))
    elif eda_type == "ri":
        for _ in range(num_new_per_technique):
            a_words = random_insertion_ver2(words, n, min_score)
            a_words.append("?")
            augmented_sentences.append(' '.join(a_words))
    elif eda_type == "rs":
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n)
            a_words.append("?")
            augmented_sentences.append(' '.join(a_words))
    elif eda_type == "rd":
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, n)
            a_words.append("?")
            augmented_sentences.append(' '.join(a_words))
    else:
        print("Wrong eda_type, choose in {sr, ri, rs, rd}")
        return augmented_sentences.append(sentence)
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    #append the original sentence
    augmented_sentences.append(sentence+"?")
    return augmented_sentences

def sentence_rs(sentences,n,num_aug):
    contexts = []
    if len(sentences)==1:
        return contexts
    while len(contexts)<num_aug:
        i = n
        new_sentences = sentences[:]
        while i>0:
            i-=1
            a = random.randrange(len(sentences))
            if a==0:
                b=1
            else:
                b = a-1
            temp = new_sentences[a]
            new_sentences[a]=new_sentences[b]
            new_sentences[b]=temp
        contexts.append(' '.join(new_sentences))
    return contexts
    
def sentence_rd(sentences,except_list, n, num_aug):
    contexts = []
    if len(except_list)==len(sentences):
        return contexts
    while len(contexts)<num_aug:
        i = n
        new_sentences = sentences[:]
        while i>0:
            i-=1
            a = random.randrange(len(new_sentences))
            while a in except_list:
                a = random.randrange(len(sentences))
            new_sentences.pop(a)
        contexts.append(' '.join(new_sentences))
    return contexts

def sentence_eda(context, orig_answer_text, eda_type, alpha=0.1, num_aug=2):
    sentences = context.split('다. ')
    new_sentences = [sentence+'다.' for sentence in sentences[:-1]]
    new_sentences.append(sentences[-1])
    answer_start = context.find(orig_answer_text)
    answer_end = answer_start+len(orig_answer_text)-1
    p = 0
    except_list=set()
    for i,st in enumerate(sentences):
        p+=len(st)
        p+=1
        if answer_start<p:
            except_list.add(i)
            break
    for i,st in enumerate(sentences):
        p+=len(st)
        p+=1
        if answer_end<p:
            except_list.add(i)
            break
    n = max(1,int(alpha*len(sentences)))
    if eda_type=="st_rs":
        new_contexts = sentence_rs(new_sentences,n,num_aug)
    elif eda_type=="st_rd":
        new_contexts = sentence_rd(new_sentences,except_list,n,num_aug)
        
    new_examples=[[new_context,new_context.find(orig_answer_text)] for new_context in new_contexts]

    return new_examples
