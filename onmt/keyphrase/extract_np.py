# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

import tqdm

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import spacy
spacy_nlp = spacy.load('en_core_web_sm')
from spacy.symbols import NOUN, PROPN, PRON

import re
import numpy as np
import spacy
import nltk

# base_dir = '/zfs1/pbrusilovsky/rum20/sum/newssum/'
# nltk.data.path.append('%s/tools/nltk/' % base_dir)
stemmer = nltk.stem.porter.PorterStemmer()
stopword_set = set(nltk.corpus.stopwords.words('english'))
stopword_set.update(['\'s', 'doe', 'n\'t', 'and', 'also', 'whether'])


def get_noun_chunks(text, trim_punct=True, remove_stopword=True):
    spacy_doc = spacy_nlp(text, disable=["textcat"])
    np_chunks = list(spacy_doc.noun_chunks)
    np_str_list = []
    for chunk in np_chunks:
        np = []
        for w in chunk:
            w = w.text
            if trim_punct:
                w = w.strip(r"""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~""")
            if remove_stopword:
                if w.lower() in stopword_set:
                    continue
            np.append(w)
        if len(np) > 0:
            np_str_list.append(' '.join(np))

    return np_str_list


def get_all_np(text, stem=True, return_set=True):
    # code to recursively combine nouns
    # 'We' is actually a pronoun but included in your question
    # hence the token.pos_ == "PRON" part in the last if statement
    # suggest you extract PRON separately like the noun-chunks above

    doc = spacy_nlp(text)
    index = 0
    nounIndices = []
    for token in doc:
        # print(token.text, token.pos_, token.dep_, token.head.text)
        if token.pos_ == 'NOUN':
            nounIndices.append(index)
        index = index + 1

    #     print(nounIndices)
    np_str_list = []

    #     for nc in doc.noun_chunks:
    #         for np in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i+1]]:
    #             print(np.text)
    # #             np_str_list.append(np)
    #             np_str_list.append(' '.join([stemmer.stem(w) for w in np.text.split()]))

    for idxValue in nounIndices:
        doc = spacy_nlp(text)
        span = doc[doc[idxValue].left_edge.i: doc[idxValue].right_edge.i + 1]
        span.merge()

        for token in doc:
            if token.dep_ == 'dobj' or token.dep_ == 'pobj' or token.pos_ == "PRON":
                #                 print(' '.join([stemmer.stem(w) for w in token.text.split()]))
                if stem:
                    np_str_list.append(' '.join([stemmer.stem(w) for w in token.text.split()]))
                else:
                    np_str_list.append(token.text)

    if return_set:
        np_str_list = set(np_str_list)

    return np_str_list


def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]


def if_present_duplicate_phrases(src_seq, tgt_seqs, stemming=True, lowercase=True):
    """
    Check if each given target sequence verbatim appears in the source sequence
    :param src_seq:
    :param tgt_seqs:
    :param stemming:
    :param lowercase:
    :param check_duplicate:
    :return:
    """
    if lowercase:
        src_seq = [w.lower() for w in src_seq]
    if stemming:
        src_seq = stem_word_list(src_seq)

    present_indices = []
    present_flags = []
    duplicate_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for tgt_seq in tgt_seqs:
        if lowercase:
            tgt_seq = [w.lower() for w in tgt_seq]
        if stemming:
            tgt_seq = stem_word_list(tgt_seq)

        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_phrase(src_seq, tgt_seq)

        # if it reaches the end of source and no match, means it doesn't appear in the source
        present_flags.append(match_flag)
        present_indices.append(match_pos_idx)

        # check if it is duplicate
        if '_'.join(tgt_seq) in phrase_set:
            duplicate_flags.append(True)
        else:
            duplicate_flags.append(False)
        phrase_set.add('_'.join(tgt_seq))

    assert len(present_flags) == len(present_indices)

    return np.asarray(present_flags), \
           np.asarray(present_indices), \
           np.asarray(duplicate_flags)


def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """

    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_flag = False
    match_pos_idx = -1
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
            break

    return match_flag, match_pos_idx


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


def all_nested_NPs(span):
    i = 0
    for i, word in enumerate(span):
        if word.pos != 89: # not a DET
            break

    span = span[i: ]
    nested_nps = []

    # a two-layer loop to get all possible nested phrases
    for k in range(1, len(span) + 1):
        for i in range(len(span) - k + 1):
            # print(span[i: i + k])
            np = span[i: i + k]
            nested_nps.append(np)

    return nested_nps


def noun_chunks(doc, remove_duplicate=True):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    noun_chunk_list = []
    labels = ['nsubj', 'dobj', 'nsubjpass', 'pcomp', 'pobj', 'dative', 'appos',
              'attr', 'ROOT']
    id2name = {tid: t for tid, t in enumerate(spacy.symbols.NAMES)}
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add('conj')
    np_set = set()

    for i, word in enumerate(doc):
        # print(i, word.text, id2name[word.pos], id2name[word.dep] if word.dep in id2name else 'np_dep')
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        # Prevent nested chunks from being produced
        if word.dep in np_deps:
            # print(doc[word.left_edge.i: word.i+1])
            # print([id2name[t.pos] for t in doc[word.left_edge.i: word.i+1]])
            if remove_duplicate:
                for np in all_nested_NPs(doc[word.left_edge.i: word.i+1]):
                    if np.text not in np_set:
                        noun_chunk_list.append(np)
                        np_set.add(np.text)
            else:
                noun_chunk_list.extend(all_nested_NPs(doc[word.left_edge.i: word.i+1]))
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                # print(doc[word.left_edge.i: word.i + 1])
                # print([id2name[t.pos] for t in doc[word.left_edge.i: word.i+1]])
                if remove_duplicate:
                    for np in all_nested_NPs(doc[word.left_edge.i: word.i+1]):
                        if np.text not in np_set:
                            noun_chunk_list.append(np)
                            np_set.add(np.text)
                else:
                    noun_chunk_list.extend(all_nested_NPs(doc[word.left_edge.i: word.i+1]))

    return noun_chunk_list


def test_np():
    text = 'A feedback vertex set of a graph G is a set S  of its vertices such that the subgraph induced by V(G)?S is a forest. The cardinality of a minimum feedback vertex set of G  is denoted by ?(G). A graph G is 2-degenerate  if each subgraph G? of G has a vertex v  such that dG?(v)?2. In this paper, we prove that ?(G)?2n/5 for any 2-degenerate n-vertex graph G and moreover, we show that this bound is tight. As a consequence, we derive a polynomial time algorithm, which for a given 2-degenerate n-vertex graph returns its feedback vertex set of cardinality at most 2n/5.'

    print(text)
    spacy_doc = spacy_nlp(text, disable=["textcat"])
    nps = list(noun_chunks(spacy_doc, remove_duplicate=True))

    print(len(nps))
    for np in nps:
        print(np.text)


def check_NP_recallM():

    datasets = ['duc', 'inspec', 'krapivin', 'nus', 'semeval', 'kp20k_valid2k', 'kp20k']

    for dataset in datasets:
        input_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/data/keyphrase/json/%s/%s_test.json' % (dataset, dataset)
        # input_path = '/Users/memray/project/kp/OpenNMT-kpg/data/keyphrase/json/%s/%s_test.json' % (dataset, dataset)
        # output_path = '/Users/memray/project/kp/OpenNMT-kpg/data/keyphrase/json/%s/%s_test_spacynp.json' % (dataset, dataset)

        input_json = open(input_path, 'r')
        # output_json = open(output_path, 'w')

        present_tgt_num_list, absent_tgt_num_list = [], []
        np_num_list = []
        recall_list = []

        for l in tqdm.tqdm(input_json):
            doc = json.loads(l)

            src_seq = meng17_tokenize(doc["title"] + ' . ' + doc["abstract"])
            stemmed_src = stem_word_list(src_seq)
            tgt_seqs = [meng17_tokenize(t) for t in doc["keywords"].lower().split(';')]
            stemmed_tgt_seqs = [stem_word_list(p) for p in tgt_seqs]

            present_tgt_flags, _, _ = if_present_duplicate_phrases(stemmed_src, stemmed_tgt_seqs)
            stemmed_present_tgts = [tgt for tgt, present in zip(stemmed_tgt_seqs, present_tgt_flags) if present]
            stemmed_absent_tgts = [tgt for tgt, present in zip(stemmed_tgt_seqs, present_tgt_flags) if not present]
            present_tgts_set = set(' '.join(p) for p in stemmed_present_tgts)

            present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
            absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if not present]

            # np_set = get_all_np(' '.join(src_seq))

            spacy_doc = spacy_nlp(' '.join(src_seq), disable=["textcat"])
            spacy_nps = noun_chunks(spacy_doc, remove_duplicate=True)
            nps = [[t.text.lower() for t in np] for np in spacy_nps]
            stemmed_nps = [' '.join(stem_word_list(p)) for p in nps]
            np_set = set(stemmed_nps)

            match_np = [p for p in np_set if p in present_tgts_set]
            recall = len(match_np) / len(stemmed_present_tgts) if len(stemmed_present_tgts) > 0 else -1.0

            np_num_list.append(len(np_set))
            recall_list.append(recall)

            present_tgt_num_list.append(len(stemmed_present_tgts))
            absent_tgt_num_list.append(len(stemmed_absent_tgts))

            output_dict = {'src': src_seq,
                           'tgt_phrases': tgt_seqs, 'num_tgt': len(tgt_seqs),
                           'present_tgt_phrases': present_tgts, 'num_present_tgt': len(present_tgts),
                           'absent_tgt_phrases': absent_tgts, 'num_absent_tgt': len(absent_tgts),
                           'noun_phrases': nps, 'num_np': len(np_set),
                           'recall': recall}
            doc.update(output_dict)
            # output_json.write(json.dumps(doc)+'\n')

            # print('*' * 50)
            # print(src_seq)
            # print('len(tgt_seqs)= %d' % len(tgt_seqs))
            # print(tgt_seqs)
            # print('len(present_tgts)= %d' % len(present_tgts))
            # print(present_tgts)
            # print('len(np_set)= %d' % len(np_set))
            # print(np_set)
            # print('len(match_np)= %d' % len(match_np))
            # print(match_np)
            # print('recall=%.4f' % recall)
            #
            # break

        present_tgt_num_list = [n for n in present_tgt_num_list if n > 0]
        absent_tgt_num_list = [n for n in absent_tgt_num_list if n > 0]

        num_data = len(recall_list)
        recall_list = [r for r in recall_list if r > -1.0]
        print('%s, #(dp)=%d, #(present_dp)=%d, '
              'avgnum_present_pred=%.4f, avgnum_absent_pred=%.4f, '
              'avgnum_NP=%.4f, recall=%.4f' %
              (dataset, num_data, len(recall_list),
               np.mean(present_tgt_num_list), np.mean(absent_tgt_num_list),
               np.mean(np_num_list), np.mean(recall_list)))
        # output_json.close()


def check_model_recallM():
    one2one_eval_paths = [
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_46000-duc-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_50000-inspec-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_86000-krapivin-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_26000-nus-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_20000-semeval-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_26000-kp20k_valid2k-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/eval/kp20k-meng17-one2one-rnn-BS128-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Covtrue-Contboth-IF1_step_36000-kp20k-exhaustive.json"
        ]

    one2seq_eval_paths = [
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_65000-duc-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_25000-inspec-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_80000-krapivin-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_45000-nus-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_55000-semeval-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_50000-kp20k_valid2k-exhaustive.json",
        "/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/eval/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_75000-kp20k-exhaustive.json",
        ]

    for eval_path in one2one_eval_paths:

        pred_basepath = eval_path[: eval_path.rfind('eval/')] + 'pred/'
        dataset = eval_path[eval_path.rfind('-', 0, eval_path.rfind('-')) + 1: eval_path.rfind('-')]
        model_name = eval_path[eval_path.rfind('/') + 1: eval_path.rfind(dataset) - 1]
        pred_path = os.path.join(pred_basepath, model_name, '%s.pred' % dataset)
        data_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/data/keyphrase/json/%s/%s_test.json' % (dataset, dataset)
        print(pred_path)

        tgt_num_list, pred_num_list = [], []
        present_tgt_num_list, absent_tgt_num_list = [], []
        present_pred_num_list, absent_pred_num_list = [], []
        present_recall_list, absent_recall_list = [], []

        with open(data_path, 'r') as data_json, open(pred_path, 'r') as pred_json:
            for data_line, pred_line in tqdm.tqdm(zip(data_json, pred_json), desc=dataset):
                doc_dict = json.loads(data_line)
                pred_dict = json.loads(pred_line)

                src_seq = meng17_tokenize(doc_dict["title"] + ' . ' + doc_dict["abstract"])
                stemmed_src = stem_word_list(src_seq)
                tgt_seqs = [meng17_tokenize(t) for t in doc_dict["keywords"].lower().split(';')]
                stemmed_tgt_seqs = [stem_word_list(p) for p in tgt_seqs]

                present_tgt_flags, _, _ = if_present_duplicate_phrases(stemmed_src, stemmed_tgt_seqs)
                stemmed_present_tgts = [tgt for tgt, present in zip(stemmed_tgt_seqs, present_tgt_flags) if present]
                stemmed_absent_tgts = [tgt for tgt, present in zip(stemmed_tgt_seqs, present_tgt_flags) if not present]
                present_tgts_set = set(' '.join(p) for p in stemmed_present_tgts)
                absent_tgts_set = set(' '.join(p) for p in stemmed_absent_tgts)
                present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
                absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if not present]

                pred_seqs = pred_dict['pred_sents']
                stemmed_pred_seqs = [stem_word_list(p) for p in pred_seqs]
                present_pred_flags, _, _ = if_present_duplicate_phrases(stemmed_src, stemmed_pred_seqs)
                stemmed_present_preds = [pred for pred, present in zip(stemmed_pred_seqs, present_pred_flags) if present]
                stemmed_absent_preds = [pred for pred, present in zip(stemmed_pred_seqs, present_pred_flags) if not present]
                present_preds_set = set(' '.join(p) for p in stemmed_present_preds)
                absent_preds_set = set(' '.join(p) for p in stemmed_absent_preds)
                present_preds = [pred for pred, present in zip(pred_seqs, present_pred_flags) if present]
                absent_preds = [pred for pred, present in zip(pred_seqs, present_pred_flags) if not present]

                match_present_pred = [p for p in present_preds_set if p in present_tgts_set]
                match_absent_pred = [p for p in absent_preds_set if p in absent_tgts_set]

                present_recall = len(match_present_pred) / len(stemmed_present_tgts) if len(stemmed_present_tgts) > 0 else -1
                absent_recall = len(match_absent_pred) / len(stemmed_absent_tgts) if len(stemmed_absent_tgts) > 0 else -1

                tgt_num_list.append(len(tgt_seqs))
                pred_num_list.append(len(pred_seqs))
                present_tgt_num_list.append(len(stemmed_present_tgts))
                absent_tgt_num_list.append(len(stemmed_absent_tgts))
                present_pred_num_list.append(len(present_preds_set))
                absent_pred_num_list.append(len(absent_preds_set))

                present_recall_list.append(present_recall)
                absent_recall_list.append(absent_recall)

        num_data = len(present_recall_list)
        # present_tgt_num_list = [n for n in present_pred_num_list if n > 0.0]
        # absent_tgt_num_list = [n for n in absent_tgt_num_list if n > 0.0]
        # present_pred_num_list = [n for n in present_pred_num_list if n > 0.0]
        # absent_pred_num_list = [n for n in absent_pred_num_list if n > 0.0]
        present_recall_list = [r for r in present_recall_list if r > -1.0]
        absent_recall_list = [r for r in absent_recall_list if r > -1.0]
        present_tgt_num_list = [n for n in present_tgt_num_list if n > 0]
        absent_tgt_num_list = [n for n in absent_tgt_num_list if n > 0]

        assert len(present_recall_list) == len(present_tgt_num_list)
        assert len(absent_recall_list) == len(absent_tgt_num_list)

        print('%s, #(dp)=%d, #(tgt)=%.4f, #(pred)=%.4f, \n'
              'num_present_doc=%d, avgnum_present_tgt=%.4f, avgnum_present_pred=%.4f, present_recall=%.4f, \n'
              'num_absent_doc=%d, avgnum_absent_tgt=%.4f, avgnum_absent_pred=%.4f, absent_recall=%.4f' %
              (dataset, num_data, np.mean(tgt_num_list), np.mean(pred_num_list),
               len(present_tgt_num_list), np.mean(present_tgt_num_list), np.mean(present_pred_num_list), np.mean(present_recall_list) if len(present_recall_list) > 0 else 0.0,
               len(absent_tgt_num_list), np.mean(absent_tgt_num_list), np.mean(absent_pred_num_list), np.mean(absent_recall_list) if len(absent_recall_list) > 0 else 0.0
               ))

if __name__ == '__main__':
    check_NP_recallM()
    # check_model_recallM()