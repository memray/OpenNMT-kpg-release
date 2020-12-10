# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

from transformers import RobertaTokenizer, RobertaTokenizerFast, AddedToken, convert_slow_tokenizer

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    original_tokenizer_path = '/Users/memray/project/kp/hf_vocab/roberta-base/'
    tokenizer_path = '/Users/memray/project/kp/hf_vocab/roberta-base-kp/'
    tokenizer_file = 'tokenizer.json'
    bpe_vocab = 'vocab.json'
    bpe_merges = 'merges.txt'

    sep_token = '<sep>'
    kp_special_tokens = ['<present>', '<absent>', '<category>']
    tokenizer = RobertaTokenizer(vocab_file=tokenizer_path+bpe_vocab,
                                 merges_file=tokenizer_path+bpe_merges,
                                 sep=sep_token, # doesn't work
                                 additional_special_tokens=kp_special_tokens)
    tokenizer1 = RobertaTokenizer.from_pretrained("roberta-base")

    print(tokenizer.tokenize(tokenizer.bos_token + 'what is wrong with <mask> <sep> I do not know either </s>'))
    print(tokenizer1.tokenize(tokenizer.bos_token + 'what is wrong with <mask> <sep> I do not know either </s>'))

    print('Vocab size=%d, base vocab size=%d' % (len(tokenizer), tokenizer.vocab_size))
    # <sep> must be additionally hard-coded
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    added_sep_token = AddedToken(sep_token, lstrip=False, rstrip=False)
    tokenizer.sep_token = sep_token
    tokenizer._sep_token = added_sep_token
    tokenizer.init_kwargs['sep_token'] = sep_token
    tokenizer.all_special_ids.append(sep_token_id)
    tokenizer.all_special_tokens.append(sep_token)
    tokenizer.all_special_tokens_extended.append(added_sep_token)
    tokenizer.special_tokens_map['sep_token'] = sep_token
    tokenizer.special_tokens_map_extended['sep_token'] = added_sep_token

    tokenizer.unique_no_split_tokens = tokenizer.all_special_tokens

    print(tokenizer.tokenize('<s> what is wrong with <mask> <sep> I do not know either </s> <present> <absent> <category>'))
    print(tokenizer1.tokenize('<s> what is wrong with <mask> <sep> I do not know either </s> <present> <absent> <category>'))

    # finally, both __slow_tokenizer and tokenizer_file must be set as follows
    fast_tokenizer3 = RobertaTokenizerFast.from_pretrained("roberta-base",
                                                           __slow_tokenizer=tokenizer, tokenizer_file=None,
                                                           vocab_file=tokenizer_path+bpe_vocab,
                                                           merges_file=tokenizer_path+bpe_merges)

    print(fast_tokenizer3.tokenize('<s> what is wrong with <mask> <sep> I do not know either </s> <present> <absent> <category>'))
    encoding = fast_tokenizer3.encode_plus('<s> what is wrong with <mask> <sep> I do not know either </s> <present> <absent> <category>')
    print(encoding["input_ids"])
    print(encoding.tokens())

    print()

    converted_tokenizer = convert_slow_tokenizer(tokenizer)
    encoding = converted_tokenizer.encode('<s> what is wrong with <mask> <sep> I do not know either </s> <present> <absent> <category>')
    print(encoding["input_ids"])
    print(encoding.tokens())
    print()
