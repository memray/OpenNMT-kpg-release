# -*- coding: utf-8 -*-
"""
It tokenizes src and tgt, and inserts the tokenized information into another json
A field token is appended to each token as a feature and they are splitted with a delimiter '￨'
An extension of bert_tokenize.py, supports word tokenization with Stanford CoreNLP and text labels.
Multiple-datasets support is done by OpenNMT.
From https://github.com/huggingface/pytorch-pretrained-BERT
"""
import argparse
import datetime
import json
import os
from functools import partial

import tqdm

from onmt import opts
from onmt.inputters.news_dataset import load_pretrained_tokenizer
import logging

from onmt.newssum import docutils
from onmt.utils.logging import init_logger


__author__ = 'Rui Meng'
__email__ = 'rui.meng@pitt.edu'

TOKENIZER_NAMES = ['roberta-base', 'bert-base-cased', 'word']
DATASET_TOKEN_MAP = {'cnndm': '[DATASET_CNNDM]',
                     'nyt': '[DATASET_NYT]',
                     'newsroom': '[DATASET_NEWSROOM]',
                     'xsum': '[DATASET_XSUM]',
                     'gigaword5': '[DATASET_GIGAWORD5]',
                     'newscrawl': '[DATASET_NEWSCRAWL]'
                     }

DENSITY_BIN_MAP = {'extractive': '[BIN_DENSITY_EXT]',
                     'abstractive': '[BIN_DENSITY_ABS]',
                     'mixed': '[BIN_DENSITY_MIX]',
                     'unknown': '[BIN_DENSITY_UNK]'
                     }

def init_opt():
    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--json_dir', '-json_dir', default='/export/share/rmeng/data/json/', help='Path to jsonl files.')
    parser.add_argument('--output_dir', '-output_dir', default='/export/share/rmeng/output',
                        help='The path of the output json files, final path is like /export/share/rmeng/output/bert-base-cased/tokenized/'
                             'folder name will be dataset_tgt, like cnndm_summary, and insides are train.jsonl, valid.jsonl, test.jsonl.')
    parser.add_argument('--datasets', '-datasets', type=str, nargs='+',
                        default=['cnndm', 'nyt', 'newsroom', 'xsum'], choices=['cnndm', 'nyt', 'newsroom', 'xsum'],
                        help='Specify which datasets to process. '
                             'Currently only support 6 datasets: cnndm nyt newsroom xsum gigaword5 newscrawl')
    parser.add_argument('--tokenizer', '-tokenizer', choices=TOKENIZER_NAMES, required=True, help='.')
    parser.add_argument('--partition', '-partition', type=str, choices=['train', 'valid', 'test'],
                        help='Specify which partition of dataset to process: train/test/valid/all')
    parser.add_argument('--shard_filename', '-shard_filename', type=str, help='.')
    parser.add_argument('--verbose', '-verbose', action='store_true', help='.')
    opts.pretrained_opts(parser) # @memray
    opt = parser.parse_args()

    return opt

logging.basicConfig(level=logging.INFO)


class Token():
    def __init__(self, token, field):
        self.token = token
        self.field = field


def tokenize_doc(doc, tokenizer_fn):
    # process title and mainbody
    title_tokens = tokenizer_fn(doc['title'])
    # tokenize each paragraph and put together as full text
    paragraphs = [l.strip() for l in doc['text'].split('\n') if len(l.strip()) > 0]
    sent_pars = [[s for s in docutils.sentence_split(p.strip(), model="nltk")
                  if len(s.strip()) > 0] for p in paragraphs]
    # flatten sentences to 1D, paragraphs are seperated with '\n', TODO add sentence delimiter
    sentences = []
    for sents in sent_pars:
        if len(sentences) > 0:
            sentences.append(['\n'])
        sentences.extend(sents)

    sentences_tokens = [tokenizer_fn(s) if s[0] != '\n' else ['\n'] for s in sentences]

    # concatenate paragraphs to get fulltext
    title_sents_tokens = [title_tokens] + [['\n']] + sentences_tokens

    # split summaries according to the dataset
    summary_sents = docutils.summary_sentence_segment(doc['summary'], doc['source'])
    summary_sents_tokens = [[t for t in tokenizer_fn(s)] for s in summary_sents]

    summary_tokens = []
    for p_id, p in enumerate(summary_sents_tokens):
        if p_id > 0:
            summary_tokens.append('\n')
        summary_tokens.extend(p)

    if ('description' in doc and doc['description']):
        desc_tokens = tokenizer_fn(doc['description'])
    elif 'metadata' in doc and 'description' in doc['metadata']:
        desc_tokens = tokenizer_fn(doc['metadata']['description'])
        doc['description'] = doc['metadata']['description']
    else:
        desc_tokens = None

    return title_tokens, sentences_tokens, title_sents_tokens, summary_sents_tokens, summary_tokens, desc_tokens


if __name__ == '__main__':
    opt = init_opt()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d')  # '%Y-%m-%d_%H:%M:%S'
    logger = init_logger(opt.output_dir + '/tokenize.%s.log' % (current_time))

    # determine whether to lowercase the text
    if opt.tokenizer == 'word' or '-cased' in opt.tokenizer:
        lowercase = False
    else:
        lowercase = True

    if opt.tokenizer == 'word':
        # initialize tokenizer (for testset, only word tokenization should be applied)
        tokenizer_fn = partial(docutils.word_tokenize, model="spacy", lowercase=lowercase)
    else:
        # Load pre-trained model tokenizer (vocabulary)
        pretrained_tokenizer = load_pretrained_tokenizer(opt.tokenizer, opt.cache_dir,
                                                         special_vocab_path=opt.special_vocab_path)
        tokenizer_fn = pretrained_tokenizer.tokenize

    for dataset in opt.datasets:
        if opt.shard_filename:
            input_jsonl_path = os.path.join(opt.json_dir, dataset, opt.shard_filename)
            logger.info('Tokenizing dataset [%s]. Loaded data from jsonl: %s ' % (dataset, input_jsonl_path))
            output_dir = os.path.join(opt.output_dir, opt.tokenizer, 'sharded_1000', dataset)
            output_jsonl_path = os.path.join(output_dir, opt.shard_filename)
            logger.info('Exporting tokenized data to %s' % output_jsonl_path)
        else:
            input_jsonl_path = os.path.join(opt.json_dir, dataset, '%s.jsonl' % (opt.partition))
            logger.info('Tokenizing dataset [%s - %s]. Loaded data from jsonl: %s ' % (dataset, opt.partition, input_jsonl_path))

            output_dir = os.path.join(opt.output_dir, opt.tokenizer, 'tokenized', dataset)
            output_jsonl_path = os.path.join(output_dir, opt.partition + '.jsonl')
            logger.info('Exporting tokenized data to %s' % output_jsonl_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_jsonl_path, 'w') as output_jsonl_writer:
            counter = 0
            src_lengths = []
            tgt_lengths = []

            for line in tqdm.tqdm(open(input_jsonl_path, 'r'), desc="Processing %s %s" % (dataset, opt.partition if opt.partition else opt.shard_filename)):
                counter += 1
                # print(counter)
                # if counter < 108120:
                #     continue
                doc = json.loads(line)

                # dump the processed data into this dict
                processed_doc = {}
                doc[opt.tokenizer] = processed_doc

                # word tokenization and extractive groundtruth should be preliminary for other tokenization
                if opt.tokenizer != 'word' and 'word' not in doc:
                    print('Word tokenization and extractive groundtruth should be preliminary for other tokenization')
                    raise AssertionError

                if opt.tokenizer == 'word':
                    # add description to 1st level
                    doc['description'] = doc['metadata']['description'] if dataset == 'cnndm' or dataset == 'xsum' else None
                    # tokenize text
                    title_tokens, sents_tokens, title_sents_tokens, \
                    summary_sents_tokens, summary_tokens, description_tokens \
                        = tokenize_doc(doc, tokenizer_fn)
                    processed_doc['token'] = {}
                    processed_doc['token']['title'] = title_tokens
                    processed_doc['token']['sents'] = sents_tokens
                    processed_doc['token']['summary'] = summary_tokens
                    processed_doc['token']['description'] = description_tokens

                    # extract different oracle parts. for consistency, let's use summary only as target to match
                    tgt_sents = docutils.summary_sentence_segment(doc['summary'], doc['source'])
                    tgt_text = docutils.summary_sents_to_text(summary_sents_tokens)
                    if opt.verbose:
                        print('[target]')
                        print(tgt_text)
                    # if doc['description']:
                    #     tgt_sents += [doc['description']]

                    processed_doc['oracle'] = {}

                    # oracle words
                    oracle_words = summary_tokens + description_tokens if description_tokens else summary_tokens
                    oracle_word_mask = [docutils.build_matching_mask(s, oracle_words,
                                                                     lowercase=True, stem=True,
                                                                     remove_punct=True, remove_stopwords=True)
                                        for s in title_sents_tokens]
                    processed_doc['oracle']['word'] = {}
                    processed_doc['oracle']['word']['target'] = oracle_words
                    processed_doc['oracle']['word']['mask'] = oracle_word_mask
                    extracted_text = docutils.mask_to_text(title_sents_tokens, oracle_word_mask)
                    rouge_score = docutils.calc_rouge(extracted_text, tgt_text)
                    processed_doc['oracle']['word']['oracle_text'] = extracted_text
                    processed_doc['oracle']['word']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[word]=%d' % len(oracle_words))
                        print(extracted_text)
                        print(rouge_score)

                    # oracle verbs
                    oracle_verb_tokens = [[verbs for verbs in docutils.get_verbs(s)] for s in tgt_sents]
                    oracle_verb_texts = [token['text'] for s in oracle_verb_tokens for token in s]
                    oracle_verb_mask = [docutils.build_matching_mask(s, oracle_verb_texts,
                                                                     lowercase=True, stem=True,
                                                                     remove_punct=True, remove_stopwords=True)
                                        for s in title_sents_tokens]
                    processed_doc['oracle']['verb'] = {}
                    processed_doc['oracle']['verb']['target'] = oracle_verb_tokens
                    processed_doc['oracle']['verb']['mask'] = oracle_verb_mask
                    extracted_text = docutils.mask_to_text(title_sents_tokens, oracle_verb_mask)
                    rouge_score = docutils.calc_rouge(extracted_text, tgt_text)
                    processed_doc['oracle']['verb']['oracle_text'] = extracted_text
                    processed_doc['oracle']['verb']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[Verb]=%d' % len(oracle_verb_texts))
                        print(extracted_text)
                        print(rouge_score)

                    # oracle noun phrases
                    sent_np_list = [docutils.get_noun_chunks(s) for s in tgt_sents]
                    np_text_list = [s[0] for s in sent_np_list]
                    np_chunk_list = [s[1] for s in sent_np_list]
                    oracle_np_words = [w for np_list in np_text_list for np in np_list for w in np]
                    oracle_np_mask = [docutils.build_matching_mask(s, oracle_np_words,
                                                                   lowercase=True, stem=False,
                                                                   remove_punct=True, remove_stopwords=True)
                                      for s in title_sents_tokens]
                    processed_doc['oracle']['noun_phrase'] = {}
                    processed_doc['oracle']['noun_phrase']['target'] = np_chunk_list
                    processed_doc['oracle']['noun_phrase']['mask'] = oracle_np_mask
                    extracted_text = docutils.mask_to_text(title_sents_tokens, oracle_np_mask)
                    rouge_score = docutils.calc_rouge(extracted_text, tgt_text)
                    processed_doc['oracle']['noun_phrase']['oracle_text'] = extracted_text
                    processed_doc['oracle']['noun_phrase']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[NP]=%d' % len([np for np_list in np_text_list for np in np_list]))
                        print(extracted_text)
                        print(rouge_score)

                    # oracle NERs
                    sent_ner_list = [docutils.get_NERs(s) for s in tgt_sents]
                    ner_text_list = [s[0] for s in sent_ner_list]
                    ner_chunk_list = [s[1] for s in sent_ner_list]
                    oracle_ner_words = [w for ner_list in ner_text_list for ner in ner_list for w in ner]
                    oracle_ner_masks = [docutils.build_matching_mask(s, oracle_ner_words,
                                                                     lowercase=True, stem=False,
                                                                     remove_punct=True, remove_stopwords=True)
                                        for s in title_sents_tokens]
                    processed_doc['oracle']['ner'] = {}
                    processed_doc['oracle']['ner']['target'] = ner_chunk_list
                    processed_doc['oracle']['ner']['mask'] = oracle_ner_masks
                    extracted_text = docutils.mask_to_text(title_sents_tokens, oracle_ner_masks)
                    rouge_score = docutils.calc_rouge(extracted_text, tgt_text)
                    processed_doc['oracle']['ner']['oracle_text'] = extracted_text
                    processed_doc['oracle']['ner']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[NER]=%d' % len([ner for ner_list in ner_text_list for ner in ner_list]))
                        print(extracted_text)
                        print(rouge_score)

                    # oracle bottom-up style fragments
                    oracle_bottomup_fragments, oracle_bottomup_mask = docutils.build_bottomup_mask(title_sents_tokens, summary_tokens)
                    processed_doc['oracle']['bottomup'] = {}
                    processed_doc['oracle']['bottomup']['target'] = oracle_bottomup_fragments
                    processed_doc['oracle']['bottomup']['mask'] = oracle_bottomup_mask
                    extracted_text = docutils.mask_to_text(title_sents_tokens, oracle_bottomup_mask)
                    rouge_score = docutils.calc_rouge(extracted_text, tgt_text)
                    processed_doc['oracle']['bottomup']['oracle_text'] = extracted_text
                    processed_doc['oracle']['bottomup']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[Bottom-up]=%d' % len(oracle_bottomup_fragments))
                        print(extracted_text)
                        print(rouge_score)

                    # oracle matching sentences
                    oracle_sent_ids, oracle_sentence_sent_masks, oracle_sentence_word_masks, extracted_text, rouge_score = \
                        docutils.build_oracle_sentence_mask(title_sents_tokens, summary_sents_tokens)
                    processed_doc['oracle']['sentence'] = {}
                    processed_doc['oracle']['sentence']['target'] = oracle_sent_ids
                    processed_doc['oracle']['sentence']['mask'] = oracle_sentence_word_masks
                    processed_doc['oracle']['sentence']['sentence_mask'] = oracle_sentence_sent_masks
                    processed_doc['oracle']['sentence']['oracle_text'] = extracted_text
                    processed_doc['oracle']['sentence']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[Sentence]=%d' % len(oracle_sent_ids))
                        print(extracted_text)
                        print(rouge_score)

                    # oracle LCS matching fragments
                    top_fragments_list, all_fragments_list, top_flat_fragments, all_flat_fragments, top_fragment_masks, all_fragment_masks, extracted_text, rouge_score\
                        = docutils.build_oracle_fragment_mask(title_sents_tokens, summary_sents_tokens,
                                                              ignore_punct=True, ignore_stopword=False, stemming=True)
                    processed_doc['oracle']['best_fragment'] = {}
                    processed_doc['oracle']['all_fragment'] = {}
                    processed_doc['oracle']['best_fragment']['target'] = top_fragments_list
                    processed_doc['oracle']['best_fragment']['mask'] = top_fragment_masks
                    processed_doc['oracle']['best_fragment']['oracle_text'] = extracted_text
                    processed_doc['oracle']['best_fragment']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[Top-Fragments]=%d' % len(top_flat_fragments))
                        print(extracted_text)
                        print(rouge_score)

                    processed_doc['oracle']['all_fragment']['target'] = all_fragments_list
                    processed_doc['oracle']['all_fragment']['mask'] = all_fragment_masks
                    extracted_text = docutils.mask_to_text(title_sents_tokens, all_fragment_masks)
                    rouge_score = docutils.calc_rouge(extracted_text, tgt_text)
                    processed_doc['oracle']['all_fragment']['oracle_text'] = extracted_text
                    processed_doc['oracle']['all_fragment']['oracle_rouge'] = rouge_score
                    if opt.verbose:
                        print('[All-Fragments]=%d' % len(all_flat_fragments))
                        print(extracted_text)
                        print(rouge_score)

                else:
                    # get corresponding masks from `word`
                    word_tokenized_doc = doc['word']['token']
                    title_words = word_tokenized_doc['title']
                    sents_words = word_tokenized_doc['sents']
                    summary_words = word_tokenized_doc['summary']
                    description_words = word_tokenized_doc['description']
                    title_sents_words = [title_words] + [['\n']] + sents_words

                    # tokenize summary/description
                    summary_words = [w if w != '\n' else '[SEP_SUM]' for w in summary_words]
                    if description_words:
                        description_words = [w if w != '\n' else '[SEP_SUM]' for w in description_words]
                    summary_tokens, summary_codes = docutils.words_to_subwords(pretrained_tokenizer, summary_words)
                    if description_words:
                        description_tokens, description_codes = docutils.words_to_subwords(pretrained_tokenizer, description_words)
                    else:
                        description_tokens, description_codes = None, None

                        # tokenize title/text and masks
                    oracle_masks = {k:v['mask'] for k,v in doc['word']['oracle'].items()}
                    title_sents_tokens = [s if s != ['\n'] else ['[SEP_PAR]'] for s in title_sents_words]
                    title_sents_tokens, title_sents_codes, oracle_sub_masks = \
                        docutils.wordmasks_to_subwords(pretrained_tokenizer, title_sents_words, oracle_masks)

                    processed_doc['token'] = {}
                    processed_doc['token']['title'] = title_sents_tokens[0]
                    processed_doc['token']['sents'] = title_sents_tokens[2:]
                    processed_doc['token']['title_sents'] = title_sents_tokens
                    processed_doc['token']['summary'] = summary_tokens
                    processed_doc['token']['description'] = description_tokens

                    processed_doc['code'] = {}
                    processed_doc['code']['title'] = title_sents_codes[0]
                    processed_doc['code']['sents'] = title_sents_codes[2:]
                    processed_doc['code']['title_sents'] = title_sents_codes
                    processed_doc['code']['summary'] = summary_codes
                    processed_doc['code']['description'] = description_codes

                    processed_doc['oracle_mask'] = {}
                    for k,v in oracle_sub_masks.items():
                        processed_doc['oracle_mask'][k] = v

                    processed_doc['oracle_sent_ids'] = doc['word']['oracle']['sentence']['target']
                    processed_doc['oracle_sent_mask'] = doc['word']['oracle']['sentence']['sentence_mask']

                # pass
                output_jsonl_writer.write(json.dumps(doc)+'\n')

            output_jsonl_writer.close()

