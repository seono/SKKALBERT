#-*- coding: utf-8 -*-

import re, json, glob, argparse
from gensim.corpora import WikiCorpus, Dictionary
from gensim.utils import to_unicode
import os
WIKI_REMOVE_CHARS = re.compile("'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):|\n", re.UNICODE)
WIKI_SPACE_CHARS = re.compile("(\\s|゙|゚|　|\&nbsp;)+", re.UNICODE)
EMAIL_PATTERN = re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
LESS_THAN_PATTERN = re.compile("\&lt;")
GREATER_THAN_PATTERN = re.compile("\&gt;")
WIKI_REMOVE_TOKEN_CHARS = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)
TAG_PATTERN = re.compile("<\s*[^>]*>|<\s*/\s*>", re.UNICODE)
TAG_SAVE_PATTERN = ["<table>", "<ol>", "<ul>", "<dl>"]

def sub_with_count(PATTERN, replace_text, orig_text, answer_start, answer_end):
    matchObj = re.finditer(PATTERN,orig_text)
    count=0
    end_count=0
    if answer_start>0:
        for match in matchObj:
            if match.start()<answer_start:
                count+=match.end()-match.start()-1
    matchObj = re.finditer(PATTERN,orig_text)
    if answer_end>0:
        for match in matchObj:
            if match.start()<answer_end:
                end_count+=match.end()-match.start()-1
    return re.sub(PATTERN, replace_text, orig_text), count, end_count

def preprocess_text(content, answer_start, answer_end):
    content, count, end_count = sub_with_count(EMAIL_PATTERN, ' ', content, answer_start, answer_end)  # remove email pattern
    answer_start-=count
    answer_end-=end_count
    content, count, end_count = sub_with_count(URL_PATTERN, ' ', content, answer_start, answer_end) # remove url pattern
    answer_start-=count
    answer_end-=end_count
    content, count, end_count = sub_with_count(WIKI_REMOVE_CHARS, ' ', content, answer_start, answer_end)  # remove unnecessary chars
    answer_start-=count
    answer_end-=end_count
    content, count, end_count = sub_with_count(WIKI_SPACE_CHARS, ' ', content, answer_start, answer_end)
    answer_start-=count
    answer_end-=end_count
    content, count, end_count = sub_with_count(MULTIPLE_SPACES, ' ', content, answer_start, answer_end)
    answer_start-=count
    answer_end-=end_count
    content, count, end_count = sub_with_count(LESS_THAN_PATTERN, '<', content, answer_start, answer_end)
    answer_start-=count
    answer_end-=end_count
    content, count, end_count = sub_with_count(GREATER_THAN_PATTERN, '>', content, answer_start, answer_end)
    answer_start-=count
    answer_end-=end_count
    return content, answer_start, answer_end
    """
    contents = []
    saved = []
    contents.append(content)
    #find the widest tag in TAG_SAVE_PATTERN
    min_i = 0
    while min_i > -1:
        text = contents.pop()
        min_idx = 99999
        min_i = -1
        for i, tag in enumerate(TAG_SAVE_PATTERN):
            st = text.find(tag)
            if st==-1:
                continue
            if st<min_idx:
                min_i = i
                min_idx = st
        if min_i != -1:
            tag = TAG_SAVE_PATTERN[min_i]
            ed = tagfinder(text,tag,min_idx)
            contents.append(text[:min_idx])
            contents.append(text[ed:])
            saved.append(text[min_idx:ed])
        else:
            contents.append(text)
    #split된 상태이기 때문에 contents와 saved의 value length확인하면서 answer_start 확인
    len_of_splited = 0
    for i, c in enumerate(contents):
        contents[i], count, end_count = sub_with_count(TAG_PATTERN, ' ', c, answer_start-len_of_splited, answer_end-len_of_splited)
        answer_start-=count
        answer_end-=end_count
        if i<len(saved):
            len_of_splited+=len(saved[i])+len(contents[i])
    content = ""
    for a in range(len(saved)):
        content+=contents[a]+saved[a]
    content+=contents[-1]
    content, count, end_count = sub_with_count(MULTIPLE_SPACES, ' ', content, answer_start, answer_end)
    answer_start-=count
    answer_end-=end_count
    return content, answer_start, answer_end
    """
def tagfinder(content, tag, start):
    """
    start should include open tag that you want to find the end index
    so, call tagfinder(content, tag, content.find(tag)) or (content, tag, start_index)
    """
    stack = 1
    start = start+len(tag)
    close_tag = tag[0:1]+'/'+tag[1:]
    while stack>0:
        op = content.find(tag,start)
        cl = content.find(close_tag,start)
        if op == -1:
            stack-=1
            start = cl+len(close_tag)
        elif op<cl:
            stack+=1
            start = op+len(tag)
        else:
            stack-=1
            start = cl+len(close_tag)
    return start

def remove_tag(content):
    return re.sub(TAG_PATTERN, ' ', content)
        