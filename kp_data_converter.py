# -*- coding: utf-8 -*-
"""
Original corpus is in JSON format. But OpenNMT separates data to two files by source/target (src/tgt).
Also OpenNMT preprocess is not flexible dealing with one-to-one/one-to-many data format.
Therefore this script means to do two things:
1. Separate a JSON data file to source/target files.
2. Output to one-to-one/one-to-many format.
3. Other specified preprocessing (lowercase, shuffle, filtering etc.)
"""
import argparse
import json
import os
import random
import re

import onmt.inputters.keyphrase_dataset as keyphrase_dataset
from onmt.keyphrase import utils

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"



def heuristic_filter(src_token, tgts_token, tgts_str, opt):
    '''
    tokenize and truncate data, filter examples that exceed the length limit
    :param src_tgts_pairs:
    :param tokenize:
    :return:
    '''
    print('*' * 50)
    print('len(src)=%d, len(tgt)=%d' % (len(src_token), len(tgts_token)))
    print('src: %s' % str(src_token))
    print('tgt: %s' % str(tgts_token))
    print('*' * 50)

    # SOURCE FILTER: if length of src is over/under the given length limit, discard
    if opt.max_src_seq_length and len(src_token) > opt.max_src_seq_length:
        print("INVALID: source is too long [len=%d]: \n%s" % (len(src_token), str(src_token)))
        return False, None, None
    if opt.min_src_seq_length and len(src_token) < opt.min_src_seq_length:
        print("INVALID: source is too short [len=%d]: \n%s" % (len(src_token), str(src_token)))
        return False, None, None

    filtered_tgts_str = []
    filtered_tgts_token = []

    # Go over each keyphrase and check its validity
    for tgt_token, tgt_str in zip(tgts_token, tgts_str):
        tgt_token_for_filter = utils.meng17_tokenize(tgt_str)

        # FILTER 1: if length of tgt exceeds limit, discard
        if opt.max_tgt_seq_length and len(tgt_token_for_filter) > opt.max_tgt_seq_length:
            print("\tInvalid Target: target is too long: %s (originally %s)" % (str(tgt_token), tgt_str))
            continue
        if opt.min_tgt_seq_length and len(tgt_token_for_filter) < opt.min_tgt_seq_length:
            print("\tInvalid Target: target is too short: %s (originally %s)" % (str(tgt_token), tgt_str))
            continue

        # FILTER 2: ingore all the keyphrases that contains strange punctuations, very DIRTY data!
        punc_flag = False
        puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', tgt_str)
        if len(puncts) > 0:
            print('-' * 50)
            print('Find punctuations in keyword: %s' % tgt_str)
            print('- tokens: %s' % str(tgt_token))
            punc_flag = True


        # FILTER 3: check the quality of long keyphrases (>5 words) with a heuristic rule, repeating meaningless words
        heuristic_rule_flag = False
        if len(tgt_token_for_filter) > 5:
            tgt_set = set(tgt_token_for_filter)
            if len(tgt_set) * 2 < len(tgt_token_for_filter):
                print('\t Invalid Target: heuristic_rule on long keyphrases (>5 words)')
                heuristic_rule_flag = True

        # FILTER 4: filter keywords like primary 75v05;secondary 76m10;65n30
        if (len(tgt_token_for_filter) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d', tgt_token_for_filter[0].strip())) \
                or (len(tgt_token_for_filter) > 1 and re.match(r'\d\d\w\d\d', tgt_token_for_filter[1].strip())):
            print('\tInvalid Target: matching template \d\d[a-z]\d\d: %s' % tgt_str)
            continue

        if (punc_flag or heuristic_rule_flag):
            if heuristic_rule_flag:
                print('\t Invalid Target: heuristic_rule on long keyphrases (>5 words)')
            if punc_flag:
                print('\t Invalid Target: found punctuation in keyphrases')
            continue

        filtered_tgts_str.append(tgt_str)
        filtered_tgts_token.append(tgt_token)

    # ignore the examples that have zero valid targets, for training they are no helpful
    if len(filtered_tgts_str) == 0:
        print('INVALID: found no valid targets')
        return False, None, None

    return True, filtered_tgts_token, filtered_tgts_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--src_file', '-src_file', required=True,
              help="Source JSON file of keyphrase dataset.")
    parser.add_argument('--output_path', '-output_path', required=True,
              help="The prefix for output files after preprocessing")

    # Data processing options
    parser.add_argument('--is_stack', '-is_stack', action='store_true', help='StackExchange data')
    parser.add_argument('--lower', '-lower', action='store_true', help='lowercase data')
    parser.add_argument('--filter', '-filter', action='store_true',
              help='Filter data by heuristics or not')
    parser.add_argument('--max_src_seq_length', '-max_src_seq_length', type=int, default=None,
              help="Max source text length")
    parser.add_argument('--min_src_seq_length', '-min_src_seq_length', type=int, default=None,
              help="Min source text length")
    parser.add_argument('--max_tgt_seq_length', '-max_tgt_seq_length', type=int, default=None,
              help="Max keyword length")
    parser.add_argument('--min_tgt_seq_length', '-min_tgt_seq_length', type=int, default=None,
              help="Min keyword length")
    parser.add_argument('--shuffle', '-shuffle', action='store_true', help="Shuffle data")
    parser.add_argument('--seed', '-seed', type=int, default=3435,
              help="Random seed")

    # Option relevant to keyphrase
    parser.add_argument('--tokenizer', '-tokenizer', type=str,
                        required=True, choices=['str', 'en_word', 'meng17', 'en_subword', 'en_retain_punc'],
                        help="Type of tokenization. "
                             "No matter which tokenizer is applied, the output is a string concatenated by whitespace."
                             "en_word: simply tokenized by whitespace"
                             "meng17: use the tokenization in Meng et al 2017"
                             "en_subword: use BPE"
                             "str: input string will be left as is")

    parser.add_argument('--replace_digit', '-replace_digit', action='store_true',
                        help="Whether replace all numbers to a special token <DIGIT>")
    parser.add_argument('--target_type', '-target_type', default='one2many',
              help="""Type of the target phrases.
                       Options are [one2one|one2many].
                       one2one means each pair of data contains only one target phrase; 
                       one2many means each pair of data contains multiple target phrases, 
                       which are concatenated in one string.""")
    parser.add_argument('--include_original', '-include_original', action='store_true',
                        help='Export all the original data as well')

    parser.add_argument('--report_every', '-report_every', type=int, default=10000,
              help="Report status every this many sentences")

    opt = parser.parse_args()

    print('*' * 50)
    print("Processing %s, type=%s" % (opt.src_file, opt.target_type))

    examples = []
    trg_count = 0
    valid_trg_count = 0

    for line_id, line in enumerate(open(opt.src_file, 'r')):
        if line_id + 1 % opt.report_every == 0:
            print("Processing %d" % line_id)

        json_dict = json.loads(line)
        if opt.is_stack:
            json_dict['abstract'] = json_dict['question']
            json_dict['keywords'] = json_dict['tags']
            del json_dict['question']
            del json_dict['tags']

        # may add more fields in the future, say dataset_name, keyword-specific features
        if 'id' in json_dict:
            id = json_dict['id']
        else:
            id = str(line_id)
        title = json_dict['title']
        abstract = json_dict['abstract']
        keywords = json_dict['keywords']

        # process strings
        # keywords may be a string concatenated by ';', make sure the output is a list of strings
        if isinstance(keywords, str):
            keywords = keywords.split(';')
            json_dict['keywords'] = keywords

        # remove all the abbreviations/acronyms in parentheses in keyphrases
        keywords = [re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', kw) for kw in keywords]

        if opt.lower:
            title = title.lower()
            abstract = abstract.lower()
            keywords = [k.lower() for k in keywords]

        if opt.tokenizer == "str":
            title_token = [title]
            abstract_token = [abstract]
            keywords_token = keywords
        elif opt.tokenizer == "en_word":
            title_token = title.split(' ')
            abstract_token = abstract.split(' ')
            keywords_token = [kw.split(' ') for kw in keywords]
        elif opt.tokenizer == "meng17":
            title_token = utils.meng17_tokenize(title)
            abstract_token = utils.meng17_tokenize(abstract)
            keywords_token = [utils.meng17_tokenize(kw) for kw in keywords]
        elif opt.tokenizer == "en_retain_punc":
            title_token = utils.retain_punc_tokenize(title)
            abstract_token = utils.retain_punc_tokenize(abstract)
            keywords_token = [utils.retain_punc_tokenize(kw) for kw in keywords]
        elif opt.tokenizer == "en_subword":
            raise NotImplementedError
        else:
            raise NotImplementedError

        if opt.replace_digit:
            title_token = utils.replace_numbers_to_DIGIT(title_token, k=2)
            abstract_token = utils.replace_numbers_to_DIGIT(abstract_token, k=2)
            keywords_token = [utils.replace_numbers_to_DIGIT(kw, k=2) for kw in keywords_token]

        src_token = title_token+["."]+abstract_token
        tgts_token = keywords_token

        # validate keywords
        if opt.filter:
            valid_flag, filtered_tgts_token, _ = heuristic_filter(src_token=src_token,
                                                             tgts_token=tgts_token,
                                                             tgts_str=keywords,
                                                             opt=opt)
            if not valid_flag:
                continue
            tgts_token = filtered_tgts_token

        trg_count += len(json_dict['keywords'])
        valid_trg_count += len(tgts_token)

        new_ex_list = []
        if opt.target_type == 'one2one':
            for tgt_token in tgts_token:
                ex = json_dict if opt.include_original else {}
                ex.update({
                    'id': id,
                    'src': ' '.join(src_token),
                    'tgt': ' '.join(tgt_token),
                })
                new_ex_list.append(ex)
        else:
            ex = json_dict if opt.include_original else {}
            ex.update({
                'id': id,
                'src': ' '.join(src_token),
                'tgt': [' '.join(tgt) for tgt in tgts_token] if opt.tokenizer!='str' else tgts_token,
            })
            new_ex_list.append(ex)

        examples.extend(new_ex_list)

    if opt.shuffle:
        random.seed(opt.seed)
        random.shuffle(examples)

    output_dir = opt.output_path[: opt.output_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # filename = '.' + (opt.tokenizer + ('.lower' if opt.lower else ''))
    filename = ''
    src_file= open(opt.output_path+filename+'.src', 'w')
    tgt_file= open(opt.output_path+filename+'.tgt', 'w')

    src_fields = ['id', 'title', 'abstract', 'src']
    tgt_fields = ['id', 'keywords', 'tgt']

    for ex_dict in examples:
        src_file.write(json.dumps({k: v for k, v in ex_dict.items() if k in src_fields})+'\n')
        tgt_file.write(json.dumps({k: v for k, v in ex_dict.items() if k in tgt_fields})+'\n')

    src_file.close()
    tgt_file.close()

    print("Process done")
    print("#(valid examples)=%d/%d" % (len(examples), line_id+1))
    print("#(valid trgs)=%d/%d" % (valid_trg_count, trg_count))
    print('*' * 50)