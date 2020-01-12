# -*- coding: utf-8 -*-
import copy
import json
import logging
import random
import time
from collections import Counter
from functools import partial
from itertools import starmap
from multiprocessing import pool, Pool

import six
import torch
import torchtext
from transformers import AutoTokenizer
from torchtext.data import Field, RawField

from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from tqdm import tqdm

from onmt.inputters.datareader_base import DataReaderBase
from onmt.inputters.dataset_base import _join_dicts, _dynamic_dict


class Token():
    def __init__(self, token, field):
        self.token = token
        self.field = field

def process_news_example(ex_dict, tgt_fields, tokenizer=None, tgt_weights=None,
                         add_meta_label=True, add_field_label=True,
                         has_special_vocab=False,
                         max_src_len=None, max_tgt_len=None,
                         return_type='str'):
    dataset = ex_dict['source']

    if add_meta_label or add_field_label:
        assert has_special_vocab, 'if add_meta_label or add_field_label, special_vocab must be given.'
    # if tokenizer_fn is given, tokenize text on-the-fly
    if tokenizer:
        text = ex_dict['text'].replace('\n', '[SEP_PAR]')
        ex_dict['text_tokens'] = ['[SEP_PAR]'] + tokenizer.tokenize(text)
        text_tokens = [Token(t, field='[PART_MAINBODY]') for t in ex_dict['text_tokens']]

        title = ex_dict['title'].replace('\n', ' ')
        ex_dict['title_tokens'] = tokenizer.tokenize(title)
        title_tokens = [Token(t, field='[PART_TITLE]') for t in ex_dict['title_tokens']]

        summary = ex_dict['text'].replace('\n', '[SEP_SUM]')
        ex_dict['summary_tokens'] = tokenizer.tokenize(summary)
        summary_tokens = [Token(t, field='[PART_SUMMARY]') for t in ex_dict['summary_tokens']]

        if dataset == 'cnndm' or dataset == 'xsum':
            desc = ex_dict['metadata']['description'].replace('\n', '[SEP_SUM]')
            ex_dict['desc_tokens'] = tokenizer.tokenize(desc)
            desc_tokens = [Token(t, field='[PART_DESCRIPTION]') for t in ex_dict['desc_tokens']]
        else:
            desc_tokens = None

    else:
        ex_dict['text_tokens'] = ['[SEP_PAR]']+[w if w != '\n' else '[SEP_PAR]' for w in ex_dict['text_tokens']]
        text_tokens = [Token(t, field='[PART_MAINBODY]') for t in ex_dict['text_tokens']]

        title_tokens = [w for w in ex_dict['title_tokens'] if w != '\n']
        title_tokens = [Token(t, field='[PART_TITLE]') for t in title_tokens]

        summary_tokens = [w if w != '\n' else '[SEP_SUM]' for w in ex_dict['summary_tokens']]
        summary_tokens = [Token(t, field='[PART_SUMMARY]') for t in summary_tokens]

        if 'desc_tokens' in ex_dict:
            desc_tokens = [w if w != '\n' else '[SEP_SUM]' for w in ex_dict['desc_tokens']]
            desc_tokens = [Token(t, field='[PART_DESCRIPTION]') for t in desc_tokens]
        else:
            desc_tokens = None

    # randomly select a target and use the rest as source
    copied_tgt_fields = copy.copy(tgt_fields)
    if 'description' in copied_tgt_fields and not desc_tokens:
        copied_tgt_fields.remove('description')
    if 'title' in copied_tgt_fields and not title_tokens:
        copied_tgt_fields.remove('title')
    if 'summary' in copied_tgt_fields and not summary_tokens:
        copied_tgt_fields.remove('summary')
    if len(copied_tgt_fields) == 0:
        pass

    if len(copied_tgt_fields) > 0:
        tgt_field = random.choices(copied_tgt_fields, weights=tgt_weights, k=1)[0]
    else:
        # for cases during testing, but no valid target field
        tgt_field = tgt_fields[0]

    infill_placeholder_token = Token('[PART_INFILL_PLACE]', field='[PART_INFILL_PLACE]')

    if tgt_field == 'summary':
        # remove title to avoid info-leaking during multi-dataset training
        #   changed from : src_tokens = title_tokens + sep_token + text_tokens
        src_tokens = title_tokens + text_tokens
        tgt_tokens = summary_tokens

    elif tgt_field == 'title':
        src_tokens = text_tokens
        tgt_tokens = title_tokens

    elif tgt_field == 'description' and desc_tokens:
        src_tokens = title_tokens + text_tokens
        tgt_tokens = desc_tokens

    elif tgt_field == 'randomsent':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # meta_label: prepend metadata labels to src tokens
    if add_meta_label:
        meta_tokens = get_meta_tokens(ex_dict, dataset, tgt_field)
        src_tokens = meta_tokens + src_tokens

    # RoBERTa model requires special tokens in order to work.
    if tokenizer:
        cls_token = Token(token=tokenizer.cls_token, field='[PART_METALABEL]')
        src_tokens = [cls_token] + src_tokens

    if max_src_len and len(src_tokens) > max_src_len:
        src_tokens = src_tokens[: max_src_len - 1]
        trunc_token = Token(token='[MAINBODY_TRUNCATED_END]', field='[PART_MAINBODY]')
        src_tokens.append(trunc_token)

    if max_tgt_len and len(tgt_tokens) > max_tgt_len:
        tgt_tokens = tgt_tokens[: max_tgt_len - 1]

    if return_type == 'str':
        if add_field_label:
            src = ' '.join([t.token + u'￨' + t.field for t in src_tokens]) + '\n'
        else:
            src = ' '.join([t.token for t in src_tokens]) + '\n'
        tgt = ' '.join([t.token for t in tgt_tokens]) + '\n'
        # (OpenNMT will prepend a <s> to tgt, but tgt_mask is not used at all)
        src_mask = torch.LongTensor([1] * len(src_tokens))
        tgt_mask = torch.LongTensor([1] * (len(tgt_tokens) + 1))
    else:
        # TODO, directly use tokenizer to encode
        src = torch.LongTensor([1] * len(src_tokens))
        tgt = torch.LongTensor([1] * len(src_tokens))
        src_mask = torch.LongTensor([1] * len(src_tokens))
        tgt_mask = torch.LongTensor([1] * len(tgt_tokens))

    new_ex_dict = {'src': src, 'tgt': tgt,
                   'indices': ex_dict['indices'] if 'indices' in ex_dict else None,
                    # for pretrained model
                   'src_mask': src_mask,
                   'tgt_mask': tgt_mask,
                   'src_tokens': [t.token for t in src_tokens],
                   'tgt_tokens': [t.token for t in tgt_tokens],
                   }

    return new_ex_dict


def process_tokenized_news_example(ex_dict, tokenizer_name, tokenizer,
                                   tgt_fields, tgt_weights=None,
                                   add_meta_label=True, add_field_label=False,
                                   has_special_vocab=False,
                                   max_src_len=None, max_tgt_len=None,
                                   return_type='str'):
    dataset = ex_dict['source']

    if add_meta_label or add_field_label:
        assert has_special_vocab, 'if add_meta_label or add_field_label, special_vocab must be given.'

    if tokenizer_name not in ex_dict:
        raise NotImplementedError('Tokenized data is not found in the json.')
    tokenized_data = ex_dict[tokenizer_name]['token']
    encoded_data = ex_dict[tokenizer_name]['code']

    title_sents_tokens = tokenized_data['title_sents']
    title_tokens = title_sents_tokens[0]
    summary_tokens = tokenized_data['summary']
    desc_tokens = tokenized_data['description']

    title_sents_codes = encoded_data['title_sents']
    title_codes = title_sents_codes[0]
    summary_codes = encoded_data['summary']
    desc_codes = encoded_data['description']

    if len(title_sents_tokens) == 0 or len(title_sents_codes) == 0:
        return None

    masks = ex_dict[tokenizer_name]['oracle_mask']
    oracle_sent_mask = ex_dict[tokenizer_name]['oracle_sent_mask']

    # our model has special_vocab, released BART doesn't
    if has_special_vocab:
        # replace the [PAD] to [SEP_PAR] (a preprocessing mistake)
        title_sents_tokens = [s if s!=['[PAD]'] else ['[SEP_PAR]'] for s in title_sents_tokens]
        pad_token_id = tokenizer.convert_tokens_to_ids(['[PAD]'])
        sep_token_id = tokenizer.convert_tokens_to_ids(['[SEP_PAR]'])
        title_sents_codes = [s if s!=pad_token_id else sep_token_id for s in title_sents_codes]

        # append a [SEP_PAR] to the beginning of title
        title_sents_tokens = [['[SEP_PAR]']] + title_sents_tokens
        title_sents_codes = [sep_token_id] + title_sents_codes
        masks = {k: [[0]]+v for k,v in masks.items()}
        oracle_sent_mask = [0] + oracle_sent_mask

        # add a sentence head token `[HEAD_SENT]` to all sentences
        title_sents_tokens = [s if len(s) == 1 else ['[HEAD_SENT]']+s for s in title_sents_tokens]
        senthead_token_id = tokenizer.convert_tokens_to_ids('[HEAD_SENT]')
        title_sents_codes = [s if len(s) == 1 else [senthead_token_id]+s for s in title_sents_codes]
        masks = {k: [s if len(s) == 1 else [0]+s for s in v] for k,v in masks.items()}
    else:
        # replace the [PAD] to [SEP_PAR] (a previous mistake). tokens and codes may not match
        dot_token_id = tokenizer.convert_tokens_to_ids(['.'])
        if len(title_sents_tokens) > 1 and title_sents_tokens[1] == ['[PAD]']:
            title_sents_tokens[1] = ['.']
            title_sents_codes[1] = dot_token_id
        new_title_sents_tokens = []
        new_title_sents_codes = []

        for sent_token, sent_code in zip(title_sents_tokens, title_sents_codes):
            new_sent_token, new_sent_code = [], []
            for t, c in zip(sent_token, sent_code):
                if t!='[PAD]' and c < tokenizer.vocab_size:
                    new_sent_token.append(t)
                    new_sent_code.append(c)
            assert len(new_sent_token) == len(new_sent_code), 'sentence lengths of token and code mismatch'
            if len(new_sent_token) > 0 and len(new_sent_code) > 0:
                new_title_sents_tokens.append(new_sent_token)
                new_title_sents_codes.append(new_sent_code)
        title_sents_tokens = new_title_sents_tokens
        title_sents_codes = new_title_sents_codes

    # randomly select a target and use the rest as source
    copied_tgt_dict = {k:v for k,v in zip(tgt_fields, tgt_weights)}
    if 'description' in copied_tgt_dict and not desc_tokens:
        del copied_tgt_dict['description']
    if 'title' in copied_tgt_dict and not title_tokens:
        del copied_tgt_dict['title']
    if 'summary' in copied_tgt_dict and not summary_tokens:
        del copied_tgt_dict['summary']
    if 'randomsent' in copied_tgt_dict and len(title_sents_tokens) < 4:
        del copied_tgt_dict['randomsent']
    if len(copied_tgt_dict) == 0:
        return None
    copied_tgt_fields = list(copied_tgt_dict.keys())
    copied_tgt_weights = list(copied_tgt_dict.values())

    if len(copied_tgt_fields) > 0:
        tgt_field = random.choices(copied_tgt_fields, weights=copied_tgt_weights, k=1)[0]
    else:
        # for cases during testing, but no valid target field
        tgt_field = tgt_fields[0]

    # field token: sentence position tokens
    field_tokens = []
    sent_count = 0
    for s in title_sents_tokens:
        if len(s) == 1 and s[0] == '[SEP_PAR]':
            field_tokens.append(['[SEP_PAR]'])
        else:
            sent_count = sent_count if sent_count <= 127 else 127
            field_tokens.append(['[SENT_POS_%d]' % sent_count] * len(s))
            sent_count += 1

    # determine src/tgt
    if tgt_field == 'summary':
        src_tokens = title_sents_tokens
        src_codes = title_sents_codes
        tgt_tokens = summary_tokens
        tgt_codes = summary_codes
    elif tgt_field == 'description' and desc_tokens:
        src_tokens = title_sents_tokens
        src_codes = title_sents_codes
        tgt_tokens = desc_tokens
        tgt_codes = desc_codes
    elif tgt_field == 'title':
        tgt_tokens = title_sents_tokens[1]
        tgt_codes = title_sents_codes[1]
        src_tokens = title_sents_tokens
        src_codes = title_sents_codes
        src_tokens[1] = ['[HEAD_SENT]', tokenizer.mask_token]
        src_codes[1] = [senthead_token_id, tokenizer.mask_token_id]
        field_tokens[1] = [field_tokens[1][0]] * 2
        masks = {k: [[0, 0] if si==1 else s for si, s in enumerate(v)] for k, v in masks.items()}
    elif tgt_field == 'randomsent':
        accum_len = 0
        for sentid, sent in enumerate(title_sents_tokens):
            if accum_len > max_src_len:
                break
            accum_len += len(sent)
        max_sentid = sentid
        sentid = random.randint(2, max_sentid)
        iter_count = 0
        while len(title_sents_tokens[sentid]) < 5 and iter_count < 5:
            iter_count += 1
            sentid = random.randint(2, max_sentid)
        tgt_tokens = title_sents_tokens[sentid]
        tgt_codes = title_sents_codes[sentid]
        src_tokens = title_sents_tokens
        src_codes = title_sents_codes
        src_tokens[sentid] = ['[HEAD_SENT]', tokenizer.mask_token]
        src_codes[sentid] = [senthead_token_id, tokenizer.mask_token_id]
        field_tokens[sentid] = [field_tokens[sentid][0]] * 2
        masks = {k: [[0, 0] if si==sentid else s for si, s in enumerate(v)] for k, v in masks.items()}
    else:
        raise NotImplementedError

    # meta_label: prepend metadata labels to src tokens
    if add_meta_label:
        # RoBERTa model requires special tokens in order to work.
        meta_tokens = get_meta_tokens(ex_dict, dataset, tgt_field,
                                      cls_token=tokenizer.cls_token, token_only=False)
        meta_texts = [t.token for t in meta_tokens]
        meta_fields = [t.field for t in meta_tokens]
        meta_codes = tokenizer.convert_tokens_to_ids(meta_texts)
        assert len(meta_tokens) == len(meta_codes) == len(meta_codes)

        src_tokens = [meta_texts] + src_tokens
        src_codes = [meta_codes] + src_codes
        field_tokens = [meta_fields] + field_tokens

        masks = {k: [[0] * len(meta_tokens)]+v for k,v in masks.items()}
        oracle_sent_mask = [0] + oracle_sent_mask
    else:
        src_tokens = [[tokenizer.cls_token]] + src_tokens
        src_codes = [[tokenizer.cls_token_id]] + src_codes
        field_tokens = [[tokenizer.cls_token]] + field_tokens
        masks = {k: [[0]]+v for k,v in masks.items()}
        oracle_sent_mask = [0] + oracle_sent_mask

    # add sent-level token pointing to the [SEP_PAR] prior to the oracle sentence
    oracle_sent_head_mask = [[0] * len(s) for s in src_tokens]
    if has_special_vocab:
        for sent_id, sent_mask in enumerate(oracle_sent_mask):
            if sent_mask == 1:
                # 1st token must be a `[HEAD_SENT]`
                if src_tokens[sent_id][0] != '[HEAD_SENT]':
                    print(src_tokens[sent_id])
                    pass
                # assert len(oracle_sent_head_mask[sent_id-1]) == 1
                oracle_sent_head_mask[sent_id][0] = 1
    masks['sentence_head'] = oracle_sent_head_mask

    # flatten all data before returning
    src_tokens = [t for s in src_tokens for t in s]
    src_codes = [t for s in src_codes for t in s]
    field_tokens = [t for s in field_tokens for t in s]
    field_codes = tokenizer.encode(field_tokens)
    masks = {k: [t for m in v for t in m] for k,v in masks.items()}

    len_src = len(src_tokens)
    if has_special_vocab:
        assert len_src == len(src_codes) == len(field_codes) == len(masks['word'])
    else:
        assert len_src == len(src_codes)
    assert len(tgt_tokens) == len(tgt_codes)

    # truncate scr/tgt sequences
    if max_src_len and len(src_tokens) > max_src_len:
        src_codes = src_codes[: max_src_len]
        src_tokens = src_tokens[: max_src_len]
        field_tokens = field_tokens[: max_src_len]
        field_codes = field_codes[: max_src_len]
        masks = {k: v[: max_src_len] for k,v in masks.items()}
    if max_tgt_len and len(tgt_tokens) > max_tgt_len:
        tgt_tokens = tgt_tokens[: max_tgt_len]
        tgt_codes = tgt_codes[: max_tgt_len]

    # add bos and eos to tgt_tokens/tgt_codes
    tgt_tokens = [tokenizer.bos_token] + tgt_tokens + [tokenizer.eos_token]
    tgt_codes = [tokenizer.bos_token_id] + tgt_codes + [tokenizer.eos_token_id]

    # tensorize them
    src_codes = torch.LongTensor(src_codes)
    field_codes = torch.LongTensor(field_codes)
    tgt_codes = torch.LongTensor(tgt_codes)
    masks = {'ext_' + k: torch.LongTensor(v) for k,v in masks.items()}

    src = src_codes
    tgt = tgt_codes

    src_mask = torch.LongTensor([1] * len(src_tokens))
    tgt_mask = torch.LongTensor([1] * len(tgt_tokens))

    new_ex_dict = {
        'src': src, 'tgt': tgt, 'src_field': field_codes,
        'src_length': len(src_tokens), 'tgt_length': len(tgt_tokens),
        'src_mask': src_mask, 'tgt_mask': tgt_mask,
        'src_tokens': src_tokens, 'tgt_tokens': tgt_tokens,
        'indices': ex_dict['indices'] if 'indices' in ex_dict else None,
    }
    new_ex_dict.update(masks)

    return new_ex_dict


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

def get_meta_tokens(doc, dataset_name, tgt_field, cls_token=None, token_only=False):
    # dataset label
    if dataset_name == 'cnn' or dataset_name == 'dailymail':
        dataset_name = 'cnndm'
    if dataset_name == 'newyorktimes':
        dataset_name = 'nyt'
    dataset_token = Token(DATASET_TOKEN_MAP[dataset_name], field='[PART_METALABEL]')

    # target type label
    target_token = tgt_field
    target_token = Token('[%s]' % target_token.upper(), field='[PART_METALABEL]')

    # density bin label, currently only newsroom has density_bin labels
    if dataset_name == 'newsroom':
        density_bin = doc['metadata']['density_bin']
        density_bin_token = Token(DENSITY_BIN_MAP[density_bin], field='[PART_METALABEL]')
    else:
        density_bin_token = Token(DENSITY_BIN_MAP['unknown'], field='[PART_METALABEL]')

    meta_tokens = [dataset_token, target_token, density_bin_token]

    # required by models like RoBERTa
    if cls_token:
        cls_token = Token(cls_token, field='[PART_METALABEL]')
        meta_tokens = [cls_token] + meta_tokens

    if token_only:
        meta_tokens = [t.token for t in meta_tokens]

    return meta_tokens


def process_news_examples_parallel(news_examples, tgt_fields, tgt_weights,
                                   meta_label, field_label, has_special_vocab,
                                   max_src_len=None, max_tgt_len=None,
                                   tokenizer=None, multi_process=False):
    # news_examples = list(news_examples)
    # news_examples = list(news_examples)[:1000]

    if multi_process:
        """pretrained_tokenizer seems not multi-processing safe
        error out "RuntimeError: received 0 items of ancdata" after a few examples (484~490)
        speed is slow as well ~20 it/s, single-processing is ~60 it/s
        """
        processed_list = []
        partial_fn = partial(process_news_example, tgt_fields=tgt_fields,
                             tokenizer=tokenizer, tgt_weights=tgt_weights,
                             add_meta_label=meta_label, add_field_label=field_label,
                             has_special_vocab=has_special_vocab,
                             max_src_len=max_src_len, max_tgt_len=max_tgt_len,
                             )
        with Pool(processes=4) as pool:
            for processed_ex in tqdm(pool.imap(partial_fn, news_examples),
                                          desc='Preparing src and tgt w/ multi-processing (tokenizing and field tokens)'):
                processed_list.append(processed_ex)
        """
        print('Preparing src and tgt w/ multiple processing (tokenizing and field tokens)')
        start_time = time.clock()
        pool = Pool(1)
        processed_list = pool.map(partial(process_news_example, tgt_fields=tgt_fields,
                                          tokenizer=tokenizer, tgt_weights=tgt_weights,
                                          add_meta_label=meta_label, add_field_label=field_label),
                                  news_examples)
        pool.close()
        end_time = time.clock()
        print("Process finished, elapsed time=%.4f, speed=%.2f it/s" % (end_time-start_time,
                                                                        len(processed_list)/(end_time-start_time)))
        """
    else:
        processed_list = [process_news_example(ex, tgt_fields=tgt_fields,
                                               tokenizer=tokenizer, tgt_weights=tgt_weights,
                                               add_meta_label=meta_label, add_field_label=field_label,
                                               max_src_len=max_src_len, max_tgt_len=max_tgt_len,)
                          for ex in tqdm(news_examples,
                                         desc='Preparing src and tgt w/ single processing (tokenizing and field tokens)')]

    new_processed_list = []
    for didx, d in enumerate(processed_list):
        # filter out None items
        if not d:
            print('Error when loading data point %d, skip for now' % didx)
            continue
        d['indices'] = didx
        new_processed_list.append(d)

    return new_processed_list


def load_tokenized_news_examples(news_examples, tokenizer_name, tokenizer,
                                 tgt_fields, tgt_weights,
                                 meta_label, field_label,
                                 has_special_vocab,
                                 max_src_len=None, max_tgt_len=None, ):
    processed_list = []
    for ex_id, ex_dict in tqdm(enumerate(news_examples), desc='Loading tensorized src and tgt'):
        # if ex_id >= 500:
        #     break
        try:
            ex_data_dict = process_tokenized_news_example(ex_dict, tokenizer_name, tokenizer,
                                                          tgt_fields=tgt_fields, tgt_weights=tgt_weights,
                                                          add_meta_label=meta_label, add_field_label=field_label,
                                                          has_special_vocab=has_special_vocab,
                                                          max_src_len=max_src_len, max_tgt_len=max_tgt_len)
            if ex_data_dict is None:
                # logging.warning("No valid %s is found in data %d, "
                #                 "or source text is faulty, title=`%s`, len(text)=%d"
                #                 % (str(tgt_fields), ex_id, ex_dict['title'], len(ex_dict['text'])))
                continue
            ex_data_dict['indices'] = ex_id
            processed_list.append(ex_data_dict)
        except Exception as e:
            logging.error("Error while processing %d data: %s" % (ex_id, ex_dict))
            logging.getLogger().exception('Exception message: ' + str(e))
            continue
    return processed_list


def build_dynamic_dict_and_masks_parallel(read_iters, fields, boseos_added, alignment_loss, alignment_targets, multi_process=False):
    src_vocabs = []
    stemmed_src_vocabs = []
    ex_dicts = []
    if multi_process:
        partial_fn = partial(_dynamic_dict,
                             src_field=fields['src'].base_field,
                             tgt_field=fields['tgt'].base_field,
                             boseos_added=boseos_added)
        with Pool(processes=4) as pool:
            for src_ex_vocab, example in tqdm(pool.imap(partial_fn, starmap(_join_dicts, zip(*read_iters))),
                                          desc='Preparing src and tgt w/ multi-processing (tokenizing and field tokens)'):
                src_vocabs.append(src_ex_vocab)
                ex_dicts.append(example)
        """
        print('Processing news examples w/ multiple processing (building dynamic_dict)')
        start_time = time.clock()
        pool = Pool()
        processed_list = pool.map(partial(_dynamic_dict,
                                          src_field=fields['src'].base_field, tgt_field=fields['tgt'].base_field),
                                  starmap(_join_dicts, zip(*read_iters)))
        end_time = time.clock()
        src_vocabs = [i[0] for i in processed_list]
        ex_dicts = [i[1] for i in processed_list]
        print("Process finished, elapsed time=%.4f, speed=%.2f it/s" % (end_time-start_time,
                                                                        len(processed_list)/(end_time-start_time)))
        """
    else:
        for ex_dict in tqdm(starmap(_join_dicts, zip(*read_iters)), desc='Processing news examples w/ single processing (building dynamic_dict)'):
            if hasattr(fields['src'], 'base_field'):
                src_field = fields['src'].base_field
                tgt_field = fields['tgt'].base_field
            else:
                src_field = fields['src']
                tgt_field = fields['tgt']
            # this assumes src_field and tgt_field are both text
            ex_dict, src_ex_vocab, stemmed_src_ex_vocab = _dynamic_dict(
                ex_dict, src_field, tgt_field,
                boseos_added=boseos_added,
                alignment_loss=alignment_loss,
                alignment_targets=alignment_targets
            )
            src_vocabs.append(src_ex_vocab)
            stemmed_src_vocabs.append(stemmed_src_ex_vocab)
            ex_dicts.append(ex_dict)

    return ex_dicts, src_vocabs, stemmed_src_vocabs


class NewsDataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, List[Tuple[str, Field]]]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """
    def __init__(self, fields, readers, data, dirs, sort_key,
                 tokenizer=None, filter_pred=None, opt=None):
        self.sort_key = sort_key
        self.tokenizer = tokenizer
        self.opt = opt

        # build src_map/alignment no matter field is available
        can_copy = True
        boseos_added = False
        if hasattr(opt, 'special_vocab_path'):
            if opt.special_vocab_path is None or opt.special_vocab_path == 'none' or opt.special_vocab_path == 'None':
                has_special_vocab = False
            else:
                has_special_vocab = True
        else:
            has_special_vocab = False

        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

        # data is directly given if it's called from translate.py
        if opt.data_format == 'srctgt' or data:
            read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                          in zip(readers, data, dirs)]
        elif opt.data_format == 'jsonl':
            # only for cases that directly load data from json files.
            read_iters = [r.read_jsonl(dir_) for r, dir_
                          in zip(readers, dirs)]
            read_iters = read_iters[0]
            tokenizer = self.tokenizer if self.tokenizer else None
            read_iters = process_news_examples_parallel(read_iters, opt.tgt_fields, opt.tgt_weights,
                                                        opt.meta_label, opt.field_label,
                                                        has_special_vocab=has_special_vocab,
                                                        max_src_len=opt.src_seq_length_trunc,
                                                        max_tgt_len=opt.tgt_seq_length_trunc,
                                                        tokenizer=tokenizer,
                                                        multi_process=False)
            read_iters = [[d for d in read_iters if d is not None]]
        elif opt.data_format == 'jsonl_tensor':
            # text has been tokenized and tensorized in advance (but not completely)
            boseos_added = True
            read_iters = [r.read_jsonl(dir_) for r, dir_
                          in zip(readers, dirs)]
            read_iters = read_iters[0]
            tokenizer = self.tokenizer if self.tokenizer else None
            read_iters = load_tokenized_news_examples(read_iters, opt.pretrained_tokenizer,
                                                      tokenizer,
                                                      opt.tgt_fields, opt.tgt_weights,
                                                      opt.meta_label, opt.field_label,
                                                      has_special_vocab=has_special_vocab,
                                                      max_src_len=opt.src_seq_length_trunc,
                                                      max_tgt_len=opt.tgt_seq_length_trunc)
            read_iters = [[d for d in read_iters if d is not None]]
        else:
            raise NotImplementedError

        # build dynamic_dict for copynet and masks for pretrained models
        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        alignment_loss = opt.alignment_loss if hasattr(opt, 'alignment_loss') else None
        alignment_targets = opt.alignment_targets if hasattr(opt, 'alignment_targets') else None
        ex_dicts, self.src_vocabs, self.stemmed_src_vocabs = build_dynamic_dict_and_masks_parallel(read_iters, fields,
                                                                          boseos_added=boseos_added,
                                                                          alignment_loss=alignment_loss,
                                                                          alignment_targets=alignment_targets
                                                                          )

        examples = []
        for ex_dict in tqdm(ex_dicts, desc='Processing data examples'):
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])
        logging.getLogger().info("Loaded %d data examples from %s" % (len(examples), str(dirs)))

        super(NewsDataset, self).__init__(examples, fields, filter_pred)

    def reload_fields(self):
        pass

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    def load_config(self, opt):
        self.opt = opt.opt

def load_dataset_from_jsonl(fields, paths, tokenizer, opt):
    dataset = NewsDataset(
        fields,
        readers=[NewsDataReader()],
        data=None,
        dirs=paths,
        sort_key=news_sort_key,
        tokenizer=tokenizer,
        opt=opt,
    )

    return dataset

class NewsDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.
        Read from both src and tgt files.
        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}

    def read_jsonl(self, sequences, _dir=None):
        """Read keyphrase data from disk. Current supported data format is JSON only.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with KeyphraseDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        # we need to make indices be the real index of the list, so replace it with a counter
        count = 0
        for i, line in enumerate(sequences):
            try:
                # default input is a jsonl line
                line = line.decode("utf-8")
                data = json.loads(line)
            except Exception:
                # data must be a dict
                if not data or len(line.strip()) == 0 or not isinstance(data, dict):
                    continue

            # insert `indices`
            count += 1
            data['indices'] = count
            yield data


def news_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    src_len = ex.src.shape[0] if isinstance(ex.src, torch.Tensor) else len(ex.src[0]) + 2
    tgt_len = ex.tgt.shape[0] if isinstance(ex.tgt, torch.Tensor) else len(ex.tgt[0]) + 1

    if hasattr(ex, "tgt"):
        return src_len, tgt_len

    return src_len


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        # some wierd bug appears in XSum (on￨[PART_MAINBODY] 0800 555 111￨[PART_MAINBODY])
        tokens = [t.split(feat_delim) for t in tokens]
        tokens = [t for t in tokens if len(t)>layer]
        tokens = [t[layer] for t in tokens]
    return tokens


class NewsMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(NewsMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

        # added by @memray for post-feature control
        self.meta_label = False
        self.field_label = False
        self.num_meta_label = None

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        # hope to truncate batch here!!!
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.
        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def news_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        NewsMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    # change from <pad> to [PAD] and <unk> to [UNK] to be compatible with BERT
    pad = "[PAD]"
    unk = "[UNK]"
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, unk_token=unk,
            tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = NewsMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field


def update_field_vocab(field, tokenizer):
    setattr(field, 'lower', False)
    setattr(field, 'pretrained_tokenizer', tokenizer)
    setattr(field, 'bos_token', tokenizer.bos_token)
    setattr(field, 'eos_token', tokenizer.eos_token)
    setattr(field, 'pad_token', tokenizer.pad_token)
    setattr(field, 'unk_token', tokenizer.unk_token)
    if not hasattr(field, 'vocab_cls'):
        setattr(field, 'vocab_cls', torchtext.vocab.Vocab(counter=Counter()))
    setattr(field.vocab_cls, 'UNK', tokenizer.unk_token)
    if not hasattr(field, 'vocab'):
        setattr(field, 'vocab', torchtext.vocab.Vocab(counter=Counter()))

    setattr(field.vocab, 'UNK', tokenizer.unk_token)
    setattr(field.vocab, 'unk_index', tokenizer.unk_token_id)
    setattr(field.vocab, 'stoi', tokenizer.encoder)
    field.vocab.stoi.update(tokenizer.added_tokens_encoder)
    assert all([id == tid for id, (tid, _) in enumerate(tokenizer.decoder.items())])
    itos = list(tokenizer.decoder.values()) + list(tokenizer.added_tokens_decoder.values())
    setattr(field.vocab, 'itos', itos)
    setattr(field.vocab, 'freqs', Counter(tokenizer.added_tokens_encoder))

    return field


def load_pretrained_tokenizer(tokenizer_name, cache_dir, special_vocab_path=None):
    print('Loading pretrained vocabulary, dumped to %s' % cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    print('Vocab size=%d, base vocab size=%d' % (len(tokenizer), tokenizer.vocab_size))
    if special_vocab_path is not None and special_vocab_path != 'none' and special_vocab_path != 'None':
        special_tokens = [w.strip() for w in open(special_vocab_path, 'r').readlines()]
        num_added_toks = tokenizer.add_tokens(special_tokens)
        print('Added', num_added_toks, 'special tokens')
        print('Vocab size=%d, base vocab size=%d' % (len(tokenizer), tokenizer.vocab_size))
    else:
        print('Special token vocab is not provided.')

    return tokenizer
