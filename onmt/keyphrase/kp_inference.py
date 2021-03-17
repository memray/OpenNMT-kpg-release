# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import copy
import json

from onmt.constants import ModelTask
from onmt.keyphrase.eval import eval_and_print
from onmt.keyphrase.pke.utils import compute_document_frequency
from onmt.utils.parse import ArgumentParser

exec('from __future__ import unicode_literals')

import os
import sys
import random

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('../onmt'))
if module_path not in sys.path:
    sys.path.append(module_path)

from onmt.translate.translator import build_translator

from kp_gen_eval_transfer import _get_parser
import string
import onmt.keyphrase.pke as pke

from nltk.corpus import stopwords
stoplist = stopwords.words('english')
import spacy
spacy_nlp = spacy.load('en_core_web_sm')

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def extract_bartkp(ex_dict):
    # Supervised Deep Keyphrase Model, using OpenNMT 2.x pipeline
    parser = _get_parser()
    config_path = '/zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer/config/transfer_kp/infer/keyphrase-one2seq-controlled.yml'
    opt = parser.parse_args('-config %s' % (config_path))

    ckpt_path = '/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp/bart_kppretrain_wiki_1e5/ckpts/checkpoint_step_100000.pt'
    opt.__setattr__('models', [ckpt_path])
    opt.__setattr__('fairseq_model', True)
    opt.__setattr__('encoder_type', 'bart')
    opt.__setattr__('decoder_type', 'bart')
    opt.__setattr__('pretrained_tokenizer', True)
    opt.__setattr__('copy_attn', False)

    opt.__setattr__('valid_batch_size', 1)
    opt.__setattr__('batch_size_multiple', 1)
    opt.__setattr__('bucket_size', 128)
    opt.__setattr__('pool_factor', 256)

    opt.__setattr__('beam_size', 1)
    opt.__setattr__('gpu', 0)

    if isinstance(opt.data, str): setattr(opt, 'data', json.loads(opt.data.replace('\'', '"')))
    setattr(opt, 'data_task', ModelTask.SEQ2SEQ)
    ArgumentParser._get_all_transform(opt)

    translator = build_translator(opt, report_score=False)

    num_pres, num_header, num_cat, num_seealso, num_infill = 5, 5, 5, 2, 0

    control_prefix = '<present>%d<header>%d<category>%d<seealso>%d<infill>%d<s>' \
                     % (num_pres, num_header, num_cat, num_seealso, num_infill)

    new_ex_dict = copy.copy(ex_dict)
    new_ex_dict['src_control_prefix'] = control_prefix

    scores, preds = translator.translate(
        src=[new_ex_dict],
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug,
        opt=opt
    )

    src_text = new_ex_dict['title'] + ' . ' + new_ex_dict['abstract']
    printout = eval_and_print(src_text, tgt_kps=ex_dict['keywords'], pred_kps=preds[0], pred_scores=scores[0])
    print(printout)


def extract_deepkp_deprecated(text_to_extract):
    # Supervised Deep Keyphrase Model, using OpenNMT 1.x pipeline
    parser = _get_parser()
    config_path = '../config/translate/config-rnn-keyphrase.yml'
    one2one_ckpt_path = '../models/keyphrase/meng17-one2one-kp20k-topmodels/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covfalse-Contboth-IF1_step_30000.pt'
    one2seq_ckpt_path = '../models/keyphrase/meng17-one2seq-kp20k-topmodels/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_50000.pt'
    opt = parser.parse_args('-config %s' % (config_path))
    setattr(opt, 'models', [one2one_ckpt_path])

    # start generating
    translator = build_translator(opt, report_score=False)
    scores, predictions = translator.translate(
        src=[text_to_extract],
        tgt=None,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug,
        opt=opt
    )
    # print results
    print('Paragraph:\n\t' + text_to_extract)
    print('Top predictions:')
    keyphrases = [kp.strip() for kp in predictions[0] if (not kp.lower().strip() in stoplist) and (kp != '<unk>')]
    for kp_id, kp in enumerate(keyphrases[: min(len(keyphrases), 20)]):
        print('\t%d: %s' % (kp_id + 1, kp.strip(string.punctuation)))


def extract_pke(text, method, dataset_path=None, df_path=None, top_k=10):
    method = method.lower()
    if method == 'tfidf':
        # 0. check if DF file exists
        if not os.path.exists(df_path):
            # stoplist for filtering n-grams
            stoplist = list(string.punctuation)

            # compute df counts and store as n-stem -> weight values
            compute_document_frequency(input_dir=dataset_path,
                                       output_file=df_path,
                                       extension='xml',  # input file extension
                                       language='en',  # language of files
                                       normalization="stemming",  # use porter stemmer
                                       stoplist=stoplist)

        # 1. create a TfIdf extractor.
        extractor = pke.unsupervised.TfIdf()

        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en_core_web_sm',
                                normalization=None)

        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        extractor.candidate_selection(n=3, stoplist=list(string.punctuation))

        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file=df_path)
        extractor.candidate_weighting(df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=top_k)
    elif method == 'yake':
        stoplist = stopwords.words('english')
        # 1. create a YAKE extractor.
        extractor = pke.unsupervised.YAKE()

        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en_core_web_sm',
                                normalization=None)

        # 3. select {1-3}-grams not containing punctuation marks and not
        #    beginning/ending with a stopword as candidates.
        extractor.candidate_selection(n=3, stoplist=stoplist)

        # 4. weight the candidates using YAKE weighting scheme, a window (in
        #    words) for computing left/right contexts can be specified.
        window = 2
        use_stems = False  # use stems instead of words for weighting
        extractor.candidate_weighting(window=window,
                                      stoplist=stoplist,
                                      use_stems=use_stems)

        # 5. get the 10-highest scored candidates as keyphrases.
        #    redundant keyphrases are removed from the output using levenshtein
        #    distance and a threshold.
        threshold = 0.8
        keyphrases = extractor.get_n_best(n=top_k, threshold=threshold)
    else:
        raise NotImplementedError


    for kp_id, kp in enumerate(keyphrases):
        print('\t%d: %s (%.4f)' % (kp_id + 1, kp[0], kp[1]))

    return keyphrases


if __name__ == '__main__':
    dataset_name = 'stackex'
    dataset_path = '/zfs1/hdaqing/rum20/kp/data/kp/json/%s/test.json' % dataset_name

    with open(dataset_path, 'r') as f:
        ex_dicts = [json.loads(l) for l in f.readlines()]
        for ex in ex_dicts:
            if dataset_name.startswith('openkp'):
                ex['title'] = ''
                ex['abstract'] = ex['text']
                ex['keywords'] = ex['KeyPhrases']
                ex['dataset_type'] = 'webpage'
            elif dataset_name.startswith('stackex'):
                ex['abstract'] = ex['question']
                ex['keywords'] = ex['tags'].split(';')
                ex['dataset_type'] = 'qa'
            elif dataset_name.startswith('kp20k') or dataset_name.startswith('duc'):
                ex['keywords'] = ex['keywords'].split(';') if isinstance(ex['keywords'], str) else ex['keywords']
                ex['dataset_type'] = 'scipaper'
            elif dataset_name.startswith('kptimes') or dataset_name.startswith('jptimes'):
                ex['keywords'] = ex['keywords'].split(';') if isinstance(ex['keywords'], str) else ex['keywords']
                ex['dataset_type'] = 'news'
            else:
                print('????')

    print('Loaded #(docs)=%d' % (len(ex_dicts)))
    doc_id = random.randint(0, len(ex_dicts))
    doc_id = 4399
    ex_dict = ex_dicts[doc_id]
    print(doc_id)

    extract_bartkp(ex_dict)
    # extract_pke(text_to_extract, method='tfidf' , dataset_path=dataset_path,
    #             df_path=os.path.abspath(dataset_path + '../%s.df.tsv.gz' % dataset_name))

