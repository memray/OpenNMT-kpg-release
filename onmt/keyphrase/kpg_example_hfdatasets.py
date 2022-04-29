# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import copy
import json
from collections import defaultdict

from onmt.constants import ModelTask
from onmt.keyphrase.eval import eval_and_print
from onmt.utils.parse import ArgumentParser

exec('from __future__ import unicode_literals')

import os
import sys
import numpy as np

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('../onmt'))
if module_path not in sys.path:
    sys.path.append(module_path)

from onmt.translate.translator import build_translator

from kp_gen_eval_transfer import _get_parser

from nltk.corpus import stopwords
stoplist = stopwords.words('english')
import spacy
spacy_nlp = spacy.load('en_core_web_sm')
import datasets

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


if __name__ == '__main__':
    #########################
    # 1. specify the config file used for inference, located at config/transfer_kp/infer/
    #   - One2One models: keyphrase-one2one.yml
    #   - One2Seq models: keyphrase-one2one.yml
    #   - wiki-pretrained models: keyphrase-one2seq-controlled.yml
    #########################
    config_path = '../../config/transfer_kp/infer/keyphrase-one2seq-controlled.yml'
    parser = _get_parser()
    opt = parser.parse_args('-config %s' % (config_path))

    # set up arguments for inference
    opt.__setattr__('use_given_inputs', True)
    opt.__setattr__('valid_batch_size', 16)
    opt.__setattr__('beam_size', 8)
    opt.__setattr__('gpu', 0) # cuda index, set -1 to use cpu

    # prefix for wiki models, to control the number of phrases of each type. Otherwise, set the prefix to empty
    num_pres, num_header, num_cat, num_seealso, num_infill = 5, 5, 5, 2, 0
    control_prefix = '<present>%d<header>%d<category>%d<seealso>%d<infill>%d<s>' \
                     % (num_pres, num_header, num_cat, num_seealso, num_infill)

    #########################
    # 2. load KPG models from pretrained checkpoints
    #    Some pretrained ckpts are available at https://huggingface.co/memray/opennmt-kpg/tree/main
    #########################
    ckpt_path = '/zfs1/pbrusilovsky/rum20/kp/openNMT-kpg-release-ckpt/opennmt-kpg-v2/wiki-pretrained/bart-wiki-step40k-bs256.checkpoint_step_40000.pt'
    IS_BART_CKPT = True
    # ckpt_path = '/zfs1/pbrusilovsky/rum20/kp/openNMT-kpg-release-ckpt/opennmt-kpg-v2/wiki-pretrained/transformer-wiki-step200k.checkpoint_step_200000.pt'
    # ckpt_path = '/zfs1/pbrusilovsky/rum20/kp/openNMT-kpg-release-ckpt/opennmt-kpg-v2/transformer-presabs-step200k/transformer_presabs_kp20k.checkpoint_step_95000.pt'
    # IS_BART_CKPT = False
    opt.__setattr__('models', [ckpt_path])

    if IS_BART_CKPT:
        opt.__setattr__('fairseq_model', True)
        opt.__setattr__('encoder_type', 'bart')
        opt.__setattr__('decoder_type', 'bart')
        opt.__setattr__('pretrained_tokenizer', True)
        opt.__setattr__('copy_attn', False)

    # initialize translator
    setattr(opt, 'data_task', ModelTask.SEQ2SEQ)
    ArgumentParser._get_all_transform(opt)
    translator = build_translator(opt, report_score=False)

    # load dataset
    dataset_name = 'midas/inspec'
    kp_dataset = datasets.load_dataset(dataset_name, name='raw', split='test')

    #########################
    # 3. start inference
    #########################
    print('Loaded #(docs)=%d' % (len(kp_dataset)))
    srcs, tgts, ex_dicts = [], [], []
    for eid, ex_dict in enumerate(kp_dataset):
        ex_dict['src_control_prefix'] = control_prefix
        src = ' '.join(ex_dict['document'])
        tgt = ex_dict['extractive_keyphrases'] + ex_dict['abstractive_keyphrases']
        ex_dict['src'] = src
        ex_dict['tgt'] = tgt
        ex_dicts.append(ex_dict)
        srcs.append(src)
        tgts.append(tgt)

    scores, preds = translator.translate(
        ex_dicts, opt=opt
    )

    #########################
    # 4. start evaluation
    #########################
    eval_results = []
    avg_scores = defaultdict(list)
    for src, tgt, score, pred in zip(srcs, tgts, scores, preds):
        printout, eval_result = eval_and_print(src, tgt_kps=tgt, pred_kps=pred, pred_scores=score, return_eval=True)
        print(printout)
        eval_results.append(eval_result)
        for gname, group in eval_result.items():
            for metric, score in group.items():
                avg_scores[f'{gname}-{metric}'].append(score)

    print("\n =======================================================")
    print(f'Summary of scores on {dataset_name}')
    for metric, scores in avg_scores.items():
        print('\t{}\t=\t\t{:.4f}'.format(metric, np.mean(scores)))
    print("\n =======================================================")
