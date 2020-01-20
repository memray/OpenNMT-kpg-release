import argparse
import datetime
import json
import os
from functools import partial
import tqdm
import numpy as np
import string

import sys
sys.path.append("/home/yingyi/Documents/OpenNMT-kpg")
print("python  path",sys.path)
from onmt.utils.logging import init_logger
from onmt import opts
from onmt.newssum import docutils
from onmt.inputters.news_dataset import load_pretrained_tokenizer
import random
import re
from stanfordcorenlp import StanfordCoreNLP
import nltk
nltk.download('stopwords')

TOKENIZER_NAMES = ['roberta-base', 'bert-base-uncased', 'word']
stemmer = nltk.stem.porter.PorterStemmer()
nlp = StanfordCoreNLP(r'/home/yingyi/Documents/tool/stanford-corenlp-full-2016-10-31')

def init_opt():
    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--json_dir', '-json_dir', default='/home/yingyi/Documents/kp20k', help='Path to jsonl files.')
    parser.add_argument('--output_dir', '-output_dir', default='/home/yingyi/Documents/output/kp20k',
                        help='The path of the output json files, final path is like /export/share/rmeng/output/bert-base-cased/tokenized/'
                             'folder name will be dataset_tgt, like cnndm_summary, and insides are train.jsonl, valid.jsonl, test.jsonl.')
    parser.add_argument('--tokenizer', '-tokenizer', default='roberta-base', choices=TOKENIZER_NAMES, help='.')
    parser.add_argument('--partition', '-partition', default='kp20k_train', type=str, choices=['kp20k_train', 'kp20k_valid', 'kp20k_test'],
                        help='Specify which partition of dataset to process: train/test/valid/all')
    parser.add_argument('--shard_filename', '-shard_filename', type=str, help='.')
    parser.add_argument('--verbose', '-verbose', action='store_true', help='.')
    parser.add_argument('--special_vocab_path', '-opt.special_vocab_path', default=None, action='store_true', help='.')

    opt = parser.parse_args()

    return opt

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

def start_end_re(match_position_idxs_key, match_pos_ends_key, keywords_exist, posi_dict, poss, cate):
    position_lists = []
    if cate=='noun':
        for starts, ends, words, pos in zip(match_position_idxs_key, match_pos_ends_key, keywords_exist, poss):
            position_list = []
            for start, end in zip(starts, ends):
                sta = posi_dict[start][0]
                en = posi_dict[end-1][-1]
                position_list.append([sta, en])
            position_lists.append({'position': position_list, 'phrase': words, 'pos': pos})
    else:
        for starts, ends, words in zip(match_position_idxs_key, match_pos_ends_key, keywords_exist):
            position_list = []
            for start, end in zip(starts, ends):
                sta = posi_dict[start][0]
                en = posi_dict[end-1][-1]
                position_list.append([sta, en])
            position_lists.append({'position': position_list, 'phrase': words})
    return position_lists

def start_end(match_position_idxs_key, match_pos_ends_key, keywords_exist, poss, cate):
    position_lists = []
    if cate=='noun':
        for starts, ends, words, pos in zip(match_position_idxs_key, match_pos_ends_key, keywords_exist, poss):
            position_list = []
            for start, end in zip(starts, ends):
                position_list.append([start, end])
            position_lists.append({'position': position_list, 'phrase': words, 'pos': pos})
    else:
        for starts, ends, words in zip(match_position_idxs_key, match_pos_ends_key, keywords_exist):
            position_list = []
            for start, end in zip(starts, ends):
                position_list.append([start, end])
            position_lists.append({'position': position_list, 'phrase': words})
    return position_lists

def prepend_space_to_words(words):
    new_words = []

    # prepend a space for all non-head and non-punctuation words
    for word in words:
        if len(new_words) == 0 or (len(word) == 1 and word in string.punctuation + string.whitespace):
            new_words.append(word)
        else:
            new_words.append(' ' + word)

    return new_words


def position_dict(position, subwords, length):
    posi_dict = [i for i in range(length, length+len(subwords))]
    return posi_dict

def words_to_subwords(tokenizer, words, pos = None):
    all_subwords = []
    all_codes = []
    all_pos = []
    all_posi = []
    spaced_words = prepend_space_to_words(words)
    length = 0
    if pos==None:
        for i, word in enumerate(spaced_words):
            if opt.tokenizer=="roberta-base":
                subwords = tokenizer.tokenize(word, add_prefix_space=True)
            else:
                subwords = tokenizer.tokenize(word)
            codes = tokenizer.convert_tokens_to_ids(subwords)
            posi_dict = position_dict(i, subwords, length)
            length = length + len(subwords)
            all_subwords.extend(subwords)
            all_codes.extend(codes)
            all_posi.append(posi_dict)
        return all_subwords, all_codes, all_posi
    else:
        new_po = []
        i = 0
        for word, po in zip(spaced_words, pos):
            if opt.tokenizer=="roberta-base":
                subwords = tokenizer.tokenize(word, add_prefix_space=True)
            else:
                subwords = tokenizer.tokenize(word)
            codes = tokenizer.convert_tokens_to_ids(subwords)
            posi_dict = position_dict(i, subwords, length)
            length = length + len(subwords)
            for _ in range(0,len(subwords)):
                new_po.append(po)
            all_subwords.extend(subwords)
            all_codes.extend(codes)
            all_pos.extend(new_po)
            all_posi.append(posi_dict)
            new_po = []
            i=i+1
        return all_subwords, all_codes, all_pos, all_posi

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
    return match_flag, match_pos_idxs, match_pos_ends, keywords_tokenizes

def macth_word(sentence_stemmer, word_stemmer, word_origin, poss):
    match_flags = []
    match_pos_idxs = []
    match_pos_ends = []
    words_tokenize = []
    absent_toeknize = []
    exist_pos = []
    for word, origin, pos in zip(word_stemmer, word_origin, poss):
        match_flag, match_pos_idx, match_pos_end, word_tokenize = if_present_phrase(sentence_stemmer, word)
        if len(match_pos_idx)>0:
            match_flags.append(match_flag)
            match_pos_idxs.append(match_pos_idx)
            match_pos_ends.append(match_pos_end)
            words_tokenize.append(origin)
            exist_pos.append(pos)
        else:
            absent_toeknize.append(origin)
    return match_flags, match_pos_idxs, match_pos_ends, words_tokenize, absent_toeknize, exist_pos

def keyword_stemmer(keywords):
    keywords_tokenize = []
    for keyword in keywords:
        keyword_stemmer = [stemmer.stem(word.lower().strip()) for word in keyword]
        keywords_tokenize.append(keyword_stemmer)
    return keywords_tokenize

def nounchunk_stemmer(nouns):
    nounchunks_tokenize = []
    for nounchunk in nouns:
        nounchunk_stemmer = [stemmer.stem(word.lower().strip()) for word in nounchunk]
        nounchunks_tokenize.append(nounchunk_stemmer)
    return nounchunks_tokenize

def pos_judge(text, model="stanfordnlp", lowercase=False):
    #print ('pos_tag', text)
    return nlp.pos_tag(text)

def listToStr(tokens):
    sentence = ""
    for token in tokens:
        sentence = sentence+str(token)+" "
    return sentence.strip()

def tokenize_doc(doc):
    # process title and abstract
    title_tokens = meng17_tokenize(doc['title'])

    abstract_tokens = meng17_tokenize(doc['abstract'])

    all_tokens = title_tokens[:]
    all_tokens.append(".")
    for tokens in abstract_tokens:
        all_tokens.append(tokens)

    keywords_tokens = []
    for keyword in doc['keywords']:
        keyword_tokenize = meng17_tokenize(keyword.strip())
        if keyword_tokenize!=[]:
            keywords_tokens.append(keyword_tokenize)
    return title_tokens, abstract_tokens, all_tokens, keywords_tokens


def label(sentence, starts, ends):
    zero = np.zeros(len(sentence))
    for start, end in zip(starts, ends):
        for st, en in zip(start, end):
            zero[st]=1
            zero[st+1:en]=2
    return [int (i) for i in zero]

def recognise_nounchunks(tagged):
    #from montylingua-2.1/ MontyREChunker.py
    lookup=[]
    words_poss = {}
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
            only_pos =  ' '.join(
                list(map(lambda filename_dict: popd_arr[filename_dict], range(len(popd_arr)))))
            stripped_str = 'NC_' + str(random.randint(0, 1000000000))
            for stripped_dict in range(len(file1)):
                if stripped_dict in range(tagged_str[0], tagged_str[1]):
                    file1[stripped_dict] = 'bar'
                    _montylingua_arr[stripped_dict] = stripped_str
            lookup.append(cron_cleaned)
            words_poss[only_word] = only_pos
    noun_phrases = [only_word.split() for only_word in words_poss.keys()]
    pos_phrases = [only_pos.split() for only_pos in words_poss.values()]
    return lookup, noun_phrases, pos_phrases


if __name__ == '__main__':
    opt = init_opt()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d')  # '%Y-%m-%d_%H:%M:%S'
    logger = init_logger(opt.output_dir + '/tokenize.%s.log' % (current_time))

    # determine whether to lowercase the text
    if opt.tokenizer == 'word' or '-base' in opt.tokenizer:
        lowercase = False
    else:
        lowercase = False

    if opt.tokenizer == 'word':
        # initialize tokenizer (for testset, only word tokenization should be applied)
        #tokenizer_fn = partial(docutils.word_tokenize, model="spacy", lowercase=lowercase)
        tokenizer_fn = nlp
    else:
        # Load pre-trained model tokenizer (vocabulary)
        pretrained_tokenizer = load_pretrained_tokenizer(opt.tokenizer, None,
                                                         special_vocab_path=opt.special_vocab_path)
        tokenizer_fn = pretrained_tokenizer.tokenize

    if opt.shard_filename:
        input_jsonl_path = os.path.join(opt.json_dir, opt.shard_filename)
        logger.info('Tokenizing dataset. Loaded data from jsonl: %s ' % (input_jsonl_path))
        output_dir = os.path.join(opt.output_dir, opt.tokenizer, 'sharded_1000')
        output_jsonl_path = os.path.join(output_dir, opt.shard_filename)
        logger.info('Exporting tokenized data to %s' % output_jsonl_path)
    else:
        input_jsonl_path = os.path.join(opt.json_dir, '%s.json' % (opt.partition))
        logger.info(
            'Tokenizing dataset [%s]. Loaded data from jsonl: %s ' % (opt.partition, input_jsonl_path))

        output_dir = os.path.join(opt.output_dir, opt.tokenizer, 'tokenized')
        output_jsonl_path = os.path.join(output_dir, opt.partition + '7.json')
        logger.info('Exporting tokenized data to %s' % output_jsonl_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_jsonl_path, 'w') as output_jsonl_writer:
        counter = 0
        src_lengths = []
        tgt_lengths = []
        keyphrase_num = 0
        keyphrase_num_roberta = 0
        noun_num = 0
        noun_roberta_num = 0
        fw_error = open("/home/yingyi/Documents/output/kp20k/error.txt","w")
        for line in tqdm.tqdm(open(input_jsonl_path, 'r', encoding='utf-8', errors='ignore'),
                              desc="Processing %s" % (
                              opt.partition if opt.partition else opt.shard_filename)):
            counter += 1
            if counter>460000: #40000, 80000, 140000, 270000, 320000, 460000
                doc = json.loads(line)
                word_doc = {}
                roberta_doc = {}
                doc['word'] = word_doc
                doc[opt.tokenizer] = roberta_doc

                title_tokens, abstract_tokens, all_tokens, keywords_tokens = tokenize_doc(doc)
                #print(all_tokens)
                pos_sentence = pos_judge(listToStr(all_tokens))
                nounchunks, nouns, poss = recognise_nounchunks(pos_sentence)
                sentence_stemmer = [stemmer.stem(word.lower().strip()) for word in all_tokens]
                keywords_stemmer = keyword_stemmer(keywords_tokens)
                nounchunks_stemmer = nounchunk_stemmer(nouns)
                word_doc["token"] = all_tokens

                match_flags_key, match_position_idxs_key, match_pos_ends_key, keywords_exist, keyword_absent, _ = macth_word(
                            sentence_stemmer, keywords_stemmer, keywords_tokens, poss)
                match_flags_noun, match_position_idxs_noun, match_pos_ends_noun, nounwords_exist, nounwords_absent, exist_poss = macth_word(
                            sentence_stemmer, nounchunks_stemmer, nouns, poss)

                word_doc['keyword_absent'] = keyword_absent
                key_positions = start_end(match_position_idxs_key, match_pos_ends_key, keywords_exist, exist_poss, 'key')
                noun_positions = start_end(match_position_idxs_noun, match_pos_ends_noun, nounwords_exist, exist_poss, 'noun')
                word_doc['keywords_position'] = key_positions
                word_doc['nouns_position'] = noun_positions

                keyphrase_num = keyphrase_num + len(key_positions)
                noun_num = noun_num + len(noun_positions)
                assert  len(exist_poss)==len(nounwords_exist)

                # tokenize text
                keywords_exist_pre = []
                keywords_absent_pre = []
                nounchunks_pre = []
                poss_re = []
                sentence_pre, _, posi_dict = words_to_subwords(pretrained_tokenizer, all_tokens)

                for words in keywords_exist:
                    keyword_pre, _, _ = words_to_subwords(pretrained_tokenizer, words)
                    keywords_exist_pre.append(keyword_pre)
                for words in keyword_absent:
                    keyword_pre, _, _ = words_to_subwords(pretrained_tokenizer, words)
                    keywords_absent_pre.append(keyword_pre)
                for words, _pos in zip(nounwords_exist, exist_poss):
                    noun_pre, _, pos_pre, _ = words_to_subwords(pretrained_tokenizer, words, pos = _pos)
                    nounchunks_pre.append(noun_pre)
                    poss_re.append(pos_pre)

                roberta_doc["token"] = sentence_pre

                roberta_doc['keyword_absent'] = keywords_absent_pre
                re_key_positions = start_end_re(match_position_idxs_key, match_pos_ends_key, keywords_exist_pre, posi_dict, exist_poss, 'key')
                re_noun_positions = start_end_re(match_position_idxs_noun, match_pos_ends_noun, nounchunks_pre, posi_dict, poss_re, 'noun')

                keyphrase_num_roberta = keyphrase_num_roberta + len(re_key_positions)
                noun_roberta_num = noun_roberta_num + len(re_noun_positions)

                roberta_doc["keywords_position"] = re_key_positions
                roberta_doc["nouns_position"] = re_noun_positions

                output_jsonl_writer.write(json.dumps(doc)+'\n')
                #print (keyphrase_num, keyphrase_num_roberta, noun_num, noun_roberta_num)
                #print (doc)


        print (keyphrase_num)
        print (keyphrase_num_roberta)
        print (noun_num)
        print (noun_roberta_num)



