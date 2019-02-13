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

from onmt.inputters.keyphrase_dataset import copyseq_tokenize

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"



def heuristic_filter(src, tgts, opt):
    '''
    tokenize and truncate data, filter examples that exceed the length limit
    :param src_tgts_pairs:
    :param tokenize:
    :return:
    '''
    src_tokens = copyseq_tokenize(src)
    # SOURCE FILTER: if length of src is over/under the given length limit, discard
    if opt.max_src_seq_length and len(src_tokens) > opt.max_src_seq_length:
        print("INVALID: source is too long: \n%s" % src)
        return False, None
    if opt.min_src_seq_length and len(src_tokens) < opt.min_src_seq_length:
        print("INVALID: source is too short: \n%s" % src)
        return False, None

    filtered_tgts = []
    for ori_tgt in tgts:

        tgt = ori_tgt

        # FILTER 1: remove all the abbreviations/acronyms in parentheses in keyphrases
        tgt = re.sub(r'\(.*?\)', '', tgt)
        tgt = re.sub(r'\[.*?\]', '', tgt)
        tgt = re.sub(r'\{.*?\}', '', tgt)

        tgt_tokens = copyseq_tokenize(tgt)
        # FILTER 2: if length of tgt exceeds limit, discard
        if opt.max_tgt_seq_length and len(tgt_tokens) > opt.max_tgt_seq_length:
            print("INVALID: target is too long: %s (originally %s)" % (str(tgt), ori_tgt))
            continue
        if opt.min_tgt_seq_length and len(tgt_tokens) < opt.min_tgt_seq_length:
            print("INVALID: target is too short: %s (originally %s)" % (str(tgt), ori_tgt))
            continue

        # FILTER 3: ingore all the keyphrases that contains strange punctuations, very DIRTY data!
        punc_flag = False
        puncts = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', tgt)
        if len(puncts) > 0:
            print('-' * 50)
            print('Find punctuations in keyword: %s' % tgt)
            print('- tokens: %s' % str(tgt_tokens))
            punc_flag = True


        # FILTER 4: check the quality of long keyphrases (>5 words) with a heuristic rule
        heuristic_rule_flag = False
        if len(tgt_tokens) > 5:
            tgt_set = set(tgt_tokens)
            if len(tgt_set) * 2 < len(tgt_tokens):
                heuristic_rule_flag = True

        # FILTER 5: filter keywords like primary 75v05;secondary 76m10;65n30
        if (len(tgt_tokens) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d', tgt_tokens[0].strip())) or (len(tgt_tokens) > 1 and re.match(r'\d\d\w\d\d', tgt_tokens[1].strip())):
            print('INVALID: matching template \d\d[a-z]\d\d: %s' % tgt)
            continue

        if (punc_flag or heuristic_rule_flag):
            print('*' * 50)
            if heuristic_rule_flag:
                print('INVALID: heuristic_rule on long keyphrases (>5 words)')
            if punc_flag:
                print('INVALID: checking punctuation in keyphrases')
            print('len(src)=%d, len(tgt)=%d' % (len(src_tokens), len(tgt_tokens)))
            print('src: %s' % str(src))
            print('tgt: %s' % str(tgts))
            print('*' * 50)
            continue

        filtered_tgts.append(ori_tgt)

    # ignore the examples that have zero valid targets, for training they are no helpful
    if len(filtered_tgts) == 0:
        return False, None

    return True, filtered_tgts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--src_file', '-src_file', required=True,
              help="Source JSON file of keyphrase dataset.")
    parser.add_argument('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")

    # Data processing options
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
    parser.add_argument('--target_type', '-target_type', required=True,
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
    print('*' * 50)

    examples = []

    for line_id, line in enumerate(open(opt.src_file, 'r')):
        if line_id + 1 % opt.report_every == 0:
            print("Processing %d" % line_id)

        json_dict = json.loads(line)

        # may add more fields in the future, say dataset_name, keyword-specific features
        if 'id' in json_dict:
            id = json_dict['id']
        else:
            id = str(line_id)
        title = json_dict['title']
        abstract = json_dict['abstract']
        keywords = json_dict['keywords']

        if isinstance(keywords, str):
            keywords = keywords.split(';')

        if opt.lower:
            title = title.lower()
            abstract = abstract.lower()
            keywords = [k.lower() for k in keywords]

        if opt.filter:
            valid_flag, filtered_keywords = heuristic_filter(src=title + ' . ' + abstract, tgts=keywords, opt=opt)
            if not valid_flag:
                continue
            keywords = filtered_keywords

        new_ex_list = []
        if opt.target_type == 'one2one':
            for keyword in keywords:
                ex = json_dict if opt.include_original else {}
                ex.update({
                    'id': id,
                    'src': title+' . '+abstract,
                    'tgt': keyword
                })
                new_ex_list.append(ex)
        else:
            ex = json_dict if opt.include_original else {}
            ex.update({
                'id': id,
                'src': title+' . '+abstract,
                'tgt': keywords
            })
            new_ex_list.append(ex)

        examples.extend(new_ex_list)

    if opt.shuffle:
        random.seed(opt.seed)
        random.shuffle(examples)

    src_file= open(opt.save_data+'.%s.src' % opt.target_type, 'w')
    tgt_file= open(opt.save_data+'.%s.tgt' % opt.target_type, 'w')

    src_fields = ['id', 'title', 'abstract', 'src']
    tgt_fields = ['id', 'keywords', 'tgt']

    for ex_dict in examples:
        src_file.write(json.dumps({k:v for k,v in ex_dict.items() if k in src_fields})+'\n')
        tgt_file.write(json.dumps({k:v for k,v in ex_dict.items() if k in tgt_fields})+'\n')
        # src_file.write(json.dumps(ex_dict['src'])+'\n')
        # tgt_file.write(json.dumps(ex_dict['tgt'])+'\n')