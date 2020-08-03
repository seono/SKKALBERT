def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def get_morph_text(text, analyzer, check=False):
    text = text.strip()
    morph_text, morph_space = [], []
    morph_eojeol_tokens = []
    clean_eojeol_tokens = []
    prev_is_whitespace = True

    orig_eojeol_tokens = []
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                orig_eojeol_tokens.append(c)
            else:
                orig_eojeol_tokens[-1] += c
            prev_is_whitespace = False
    """
    for token in text.split(" "):
        if len(token) > 0:
            orig_eojeol_tokens.append(token)
    """
    idx = 0
    for i, [word, postag] in enumerate(analyzer.pos(text)):
        postag = postag.split("+")[0]
        morphtag_word = word + "/" + postag
        if len(word) == 0:
            continue
        morph_text.append(morphtag_word)

        if i == 0:
            morph_eojeol_tokens.append(morphtag_word)
            clean_eojeol_tokens.append(word)
            morph_space.append("B")
            continue

        if "".join(clean_eojeol_tokens[-1].split(" + ")) == orig_eojeol_tokens[idx]:
            morph_eojeol_tokens.append(morphtag_word)
            clean_eojeol_tokens.append(word)
            morph_space.append("B")
            idx += 1
        else:
            morph_eojeol_tokens[-1] += " + " + morphtag_word
            clean_eojeol_tokens[-1] += " + " + word
            morph_space.append("I")

    convert_text = []
    for token in clean_eojeol_tokens:
        convert_text.append("".join(token.split(" + ")))
    convert_text = " ".join(convert_text)
    #mecab에 없는 단어 ex. 내셔날, 메릿 등 고유명사들 분류 안됨
    mecab_error = 0
    if check and (convert_text != " ".join(orig_eojeol_tokens)):
        mecab_error = 1
    return " ".join(morph_text), morph_eojeol_tokens, morph_space, mecab_error


def remove_postag(morph_tokens):
    """
    token뒤에 고유명사/NNG 와 같은식으로 형태소태그가 붙어있으므로 제거
    """
    tokens = []
    for token in morph_tokens:
        idx = token.rfind("/")
        if idx != -1:
            word = token[:idx]
        else:
            word = token
        tokens.append(word)
    return tokens


def improve_morph_answer_span(tokens, spaces, input_start, input_end, orig_answer_text):
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = []
            for token, space in zip(tokens[new_start:(new_end+1)], spaces[new_start:(new_end+1)]):
                if len(text_span) == 0 or space == "B":
                    text_span.append(token)
                elif space == "I":
                    text_span[-1] += token
            text_span = " ".join(text_span)

            if text_span == orig_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)

def improve_answer_span(tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
            elif len(text_span)<len(tok_answer_text):
                break

    return (input_start, input_end)

def check_is_max_context(spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, span) in enumerate(spans):
        end = span.start + span.length - 1
        if position < span.start:
            continue
        if position > end:
            continue
        num_left_context = position - span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
