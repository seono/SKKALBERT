import random
from random import shuffle
random.seed(1)
from .models.word_eval import WordEmbeddingEvaluator

def WordEmbeddingModel_init(model_name):

    global embedd_model
    if model_name == "fasttext":
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/fasttext/fasttext.vec",
			vecs_bin_fname = "notebooks/embedding/data/word-embeddings/fasttext/fasttext.bin",
			method = "fasttext", dim=100, tokenizer_name="mecab")
    elif model_name == "glove":
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/glove/glove.txt",
            method="glove", dim=100, tokenizer_name="mecab")
    elif model_name == "swivel":
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/swivel/row_embedding.tsv",
			method="swivel", dim=100, tokenizer_name="mecab")
    else:
        embedd_model = WordEmbeddingEvaluator(
            vecs_txt_fname = "notebooks/embedding/data/word-embeddings/word2vec/word2vec",
            method="word2vec", dim=100, tokenizer_name="mecab")

def synonym_replacement(words, n, min_score):
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word, min_score)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word, min_score=0.7):
    synonyms = set()
    for word, score in embedd_model.most_similar(word,topn=10):
        if(score<min_score):
            break
        synonyms.add(word)
    return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

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

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, eda_type, alpha=0.1, num_aug=9, min_score=0.7):
    #remove ?
    sentence= sentence[:-1]
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n = max(1, int(alpha*num_words))

    #sr
    if eda_type == "sr":
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n, min_score)
            a_words.append("?")
            augmented_sentences.append(' '.join(a_words))
    elif eda_type == "ri":
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n, min_score)
            a_words.append("?")
            augmented_sentences.append(' '.join(a_words))
    elif eda_type == "rs":
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n)
            a_words.append("?")
            augmented_sentences.append(' '.join(a_words))
    elif eda_type == "rd":
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, alpha)
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