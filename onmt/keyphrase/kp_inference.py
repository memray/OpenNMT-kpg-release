# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import string

from onmt.keyphrase.pke.utils import compute_document_frequency

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

from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from kp_gen_eval import _get_parser
import string
import onmt.keyphrase.pke as pke

from nltk.corpus import stopwords
stoplist = stopwords.words('english')


__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def extract_deepkp(text_to_extract):
    # Supervised Deep Keyphrase Model
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
    dataset_name = 'SF_Prod'
    dataset_path = '../data/salesforce/%s/' % dataset_name
    prod_dicts = []
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            filepath = subdir + os.sep + file
            text = open(filepath, 'r').readlines()
            text = '\n'.join([l.strip() for l in text])
            doc = {'name': file, 'path': filepath, 'text': text}
            prod_dicts.append(doc)

    print('Loaded #(PROD docs)=%d' % (len(prod_dicts)))
    doc_id = random.randint(0, len(prod_dicts))
    doc = prod_dicts[doc_id]
    text_to_extract = doc['text']
    print(doc_id)
    print(doc['name'])
    print(text_to_extract)

    extract_deepkp(text_to_extract)
    extract_pke(text_to_extract, method='tfidf' , dataset_path=dataset_path,
                df_path=os.path.abspath(dataset_path + '../%s.df.tsv.gz' % dataset_name))