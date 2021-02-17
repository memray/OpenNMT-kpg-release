#!/usr/bin/python
# -*- coding: utf-8 -*-

import simplejson as json
from nltk.tokenize import sent_tokenize
import nltk
import re
from stanfordcorenlp import StanfordCoreNLP
import random
from transformers import BertTokenizer

stemmer = nltk.stem.porter.PorterStemmer()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

stanfordnlp = None


def BertTokenizer(tokens):
    bert_tokens = []
    for token in tokens:
        token_sp = tokenizer.tokenize(token)
        for t in token_sp:
            bert_tokens.append(t)
    return bert_tokens

def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """
    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_flag = False
    match_pos_idx = -1
    match_pos_idxs = []
    match_pos_ends = []
    keywords_tokenizes = []
    for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
        match_flag = True
        # iterate each word in target, if one word does not match, set match=False and break
        for seq_idx, seq_w in enumerate(phrase_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            match_pos_idxs.append(match_pos_idx)
            match_pos_ends.append(match_pos_idx+len(phrase_str_tokens))
            keywords_tokenizes.append(phrase_str_tokens)
            #break
    #print (match_pos_idxs)
    return match_flag, match_pos_idxs, match_pos_ends, keywords_tokenizes


def recognise_nounchunks(tagged):
    #from montylingua-2.1/ MontyREChunker.py
    lookup=[]
    only_words=[]
    info_dict = tagged
    file1 = list(map(lambda filename_dict: filename_dict[0], info_dict))
    _montylingua_arr = list(map(lambda filename_dict: filename_dict[1], info_dict))
    # filename_p = "((PDT )?(DT |PRP[$] |WDT |WP[$] )(VBG |VBD |VBN |JJ |JJR |JJS |, |CC |NN |NNS |NNP |NNPS |CD )*(NN |NNS |NNP |NNPS |CD )+)"
    # groupnames1 = "((PDT )?(JJ |JJR |JJS |, |CC |NN |NNS |NNP |NNPS |CD )*(NN |NNS |NNP |NNPS |CD )+)"
    # case1 = "(" + filename_p + "|" + groupnames1 + "|EX |PRP |WP |WDT )"
    filename_p = "((PDT )?(VBG |VBD |VBN |JJ |JJR |JJS |CD )*(NN |NNS |NNP |NNPS |CD )+)"
    case1 = "(" + filename_p+ ")"
    case1 = "(" + case1 + 'POS )?' + case1
    case1 = ' ' + case1
    case1 = re.compile(case1)
    awk1 = 1

    while awk1:
        awk1 = 0
        gawks = ' ' + ' '.join(_montylingua_arr) + ' '
        groupnames_str = case1.search(gawks)

        if groupnames_str:
            awk1 = 1
            info_str = len(gawks[:groupnames_str.start()].split())
            cleaned_arr = len(gawks[groupnames_str.end():].split())
            tagged_str = (info_str, len(_montylingua_arr) - cleaned_arr)
            mores = file1[tagged_str[0]:tagged_str[1]]
            popd_arr = _montylingua_arr[tagged_str[0]:tagged_str[1]]
            cron_cleaned = ' '.join(
                list(map(lambda filename_dict: mores[filename_dict] + '/' + popd_arr[filename_dict], range(len(mores)))))
            only_word =  ' '.join(
                list(map(lambda filename_dict: mores[filename_dict], range(len(mores)))))
            stripped_str = 'NC_' + str(random.randint(0, 1000000000))
            for stripped_dict in range(len(file1)):
                if stripped_dict in range(tagged_str[0], tagged_str[1]):
                    file1[stripped_dict] = 'bar'
                    _montylingua_arr[stripped_dict] = stripped_str
            lookup.append(cron_cleaned)
            only_words.append(only_word)
    return lookup, list(set(only_words))


def pos(text, model="stanfordnlp", lowercase=False):
    if model == 'stanfordnlp':
        global stanfordnlp
        if not stanfordnlp:
            nlp = StanfordCoreNLP(r'/home/yingyi/Documents/tool/stanford-corenlp-full-2016-10-31')
        return nlp.pos_tag(text)
    else:
        raise NotImplementedError


def listToStr(tokens):
    sentence = ""
    for token in tokens:
        sentence = sentence+token+" "
    return sentence.strip()

def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    return tokens

def keyword_stemmer(keywords):
    keywords_tokenize = []
    for keyword in keywords:
        keyword_tokenize = meng17_tokenize(keyword.strip())
        keyword_stemmer = [stemmer.stem(word.lower().strip()) for word in keyword_tokenize]
        keywords_tokenize.append(keyword_stemmer)
    return keywords_tokenize

def nounchunk_stemmer(nouns):
    nounchunks_tokenize = []
    for nounchunk in nouns:
        nounchunk_tokenize = meng17_tokenize(nounchunk.strip())
        nounchunk_stemmer = [stemmer.stem(word.lower().strip()) for word in nounchunk_tokenize]
        nounchunks_tokenize.append(nounchunk_stemmer)
    return nounchunks_tokenize


def macth_word(sentence_stemmer, word_stemmer):
    match_flags = []
    match_pos_idxs = []
    match_pos_ends = []
    words_tokenize = []
    for word in word_stemmer:
        match_flag, match_pos_idx, match_pos_end, word_tokenize = if_present_phrase(sentence_stemmer, word)
        print (match_pos_idx)
        match_flags.append(match_flag)
        match_pos_idxs.append(match_pos_idx)
        match_pos_ends.append(match_pos_end)
        words_tokenize.append(word_tokenize)
    return match_flags, match_pos_idxs, match_pos_ends, words_tokenize

if __name__ == '__main__':
    #fw = open("","w")
    with open('/home/yingyi/Documents/kp20k/kp20k_test.json','r') as f:
        lines = f.readlines()
        for line in lines:
            dict = json.loads(line)
            abstarct = dict['abstract']
            keywords = dict['keywords'].split(";")
            print (keywords)
            title = dict['title']
            sentence = title.strip() + ". " + abstarct.strip()
            sentence_tokenize = meng17_tokenize(sentence)
            pos_sentence = pos(listToStr(sentence_tokenize))
            nounchunks, nouns = recognise_nounchunks(pos_sentence)
            sentence_stemmer = [stemmer.stem(word.lower().strip()) for word in sentence_tokenize]
            keywords_stemmer = keyword_stemmer(keywords)
            nounchunks_stemmer = nounchunk_stemmer(nouns)


            match_flags_key, match_position_idxs_key, match_pos_ends_key, keywords_exist = macth_word(sentence_stemmer, keywords_stemmer)
            match_flags_noun, match_position_idxs_noun, match_pos_ends_noun, nounwords_exist = macth_word(sentence_stemmer, nounchunks_stemmer)

            keywords_stemmer_bert = []
            nounchunks_stemmer_bert = []
            sentence_bert = BertTokenizer(sentence_stemmer)

            for words in keywords_stemmer:
                keyword_bert = BertTokenizer(words)
                keywords_stemmer_bert.append(keyword_bert)
            for words in nounchunks_stemmer:
                noun_bert = BertTokenizer(words)
                nounchunks_stemmer_bert.append(noun_bert)
            print (keywords_stemmer_bert)
            match_flags_key, match_position_idxs_key, match_pos_ends_key, keywords_exist = macth_word(sentence_bert, keywords_stemmer_bert)
            match_flags_noun, match_position_idxs_noun, match_pos_ends_noun, nounwords_exist = macth_word(sentence_bert, nounchunks_stemmer_bert)

            print ('match_position_idxs_key', match_position_idxs_key, match_pos_ends_key)







            #keywords_tokenize, sentence_tokenize_ori, match_flags, match_pos_idxs = tokenize(sentence, keywords)





