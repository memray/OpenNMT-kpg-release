
import argparse
import json
import math
import logging
import string

import nltk
import scipy
import torch
from nltk.stem.porter import *
import numpy as np
from collections import Counter

import os

from torch.autograd import Variable

import config
from nltk.translate.bleu_score import sentence_bleu as bleu
from onmt.keyphrase.utils import Progbar
from onmt.utils.logging import init_logger

stemmer = PorterStemmer()


def process_predseqs(pred_seqs, unk_token):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :return:
    '''
    valid_flags = []

    for seq in pred_seqs:
        keep_flag = True

        if len(seq) == 0:
            keep_flag = False

        if keep_flag and any([w == unk_token for w in seq]):
            keep_flag = False

        if keep_flag and any([w == '.' or w == ',' for w in seq]):
            keep_flag = False

        valid_flags.append(keep_flag)

    return np.asarray(valid_flags)


def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """

    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
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
        src_to_match = stem_word_list(src_seq)
    else:
        src_to_match = src_seq

    present_indices = []
    present_flags = []
    duplicate_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for tgt_seq in tgt_seqs:
        if lowercase:
            tgt_to_match = [w.lower() for w in tgt_seq]
        if stemming:
            tgt_to_match = stem_word_list(tgt_to_match)
        else:
            tgt_to_match = tgt_to_match

        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_phrase(src_to_match, tgt_to_match)

        # if it reaches the end of source and no match, means it doesn't appear in the source
        present_flags.append(match_flag)
        present_indices.append(match_pos_idx)

        # check if it is duplicate
        if '_'.join(tgt_to_match) in phrase_set:
            duplicate_flags.append(True)
        else:
            duplicate_flags.append(False)
        phrase_set.add('_'.join(tgt_to_match))

    assert len(present_flags) == len(present_indices)

    return np.asarray(present_flags), \
           np.asarray(present_indices), \
           np.asarray(duplicate_flags)


def evaluate(src_list, tgt_list, pred_list, unk_token, logger=None, verbose=False):
    # progbar = Progbar(logger=logger, title='', target=len(pred_list), total_examples=len(pred_list))

    # 'k' means the number of phrases in ground-truth
    topk_range = [5, 10, 'k']
    absent_topk_range = [10, 30, 50]
    metric_names = ['precision', 'recall', 'f_score']

    score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    '''
    process each example in current batch
    '''
    for i, (src_dict, tgt_dict, pred_dict) in enumerate(zip(src_list, tgt_list, pred_list)):
        src_seq = pred_dict["src_raw"]
        tgt_seqs =[t.split() for t in tgt_dict["tgt"]]
        pred_sents = pred_dict["pred_sents"]
        pred_idxs = pred_dict["preds"]
        pred_scores = pred_dict["pred_scores"]
        # src, src_str, tgt, tgt_str_seqs, tgt_copy, pred_seq, oov
        print_out = '======================  %d =========================' % (i)
        print_out += '\n[Title]: %s \n' % (src_dict["title"])
        print_out += '[Abstract]: %s \n' % (src_dict["abstract"])
        # print_out += '[Source tokenized][%d]: %s \n' % (len(src_seq), ' '.join(src_seq))
        # print_out += 'Real Target [%d] \n\t\t%s \n' % (len(tgt_seqs), str(tgt_seqs))

        # check which phrases are present in source text
        present_tgt_flags, _, _ = if_present_duplicate_phrases(src_seq, tgt_seqs)

        print_out += '[GROUND-TRUTH] #(present)/#(all targets)=%d/%d\n' % (sum(present_tgt_flags), len(present_tgt_flags))
        print_out += '\n'.join(['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in zip(tgt_seqs, present_tgt_flags)])

        # 1st filtering, ignore phrases having <unk> and puncs
        valid_pred_flags = process_predseqs(pred_sents, unk_token)
        # 2nd filtering: if filter out phrases that don't appear in text, and keep unique ones after stemming
        present_pred_flags, _, duplicate_flags = if_present_duplicate_phrases(src_seq, pred_sents)
        # treat duplicates as invalid
        valid_pred_flags = valid_pred_flags * ~duplicate_flags
        valid_and_present = valid_pred_flags * present_pred_flags
        print_out += '\n[PREDICTION] #(valid)=%d, #(present)=%d, #(retained&present)=%d, #(all)=%d\n' % (sum(valid_pred_flags), sum(present_pred_flags), sum(valid_and_present), len(pred_sents))
        print_out += ''

        # compute match scores (exact, partial and mixed), for exact it's a list otherwise matrix
        match_scores_exact = get_match_result(true_seqs=tgt_seqs, pred_seqs=pred_sents, type='exact')
        match_scores_partial = get_match_result(true_seqs=tgt_seqs, pred_seqs=pred_sents, type='ngram')
        # simply add full-text to n-grams might not be good as its contribution is not clear
        match_scores_mixed = get_match_result(true_seqs=tgt_seqs, pred_seqs=pred_sents, type='mixed')

        # sanity check of pred
        num_pred = len(pred_dict["preds"])
        for d in [pred_idxs, pred_sents, pred_scores,
                  match_scores_exact, valid_pred_flags,
                  present_pred_flags, pred_dict["copied_flags"]]:
            assert len(d) == num_pred

        '''
        Print and export predictions
        '''
        preds_out = ''
        for p_id, (pred_idx, word, score,
                   match, match_soft,
                   is_valid, is_present, copied_flag) in enumerate(
                zip(pred_idxs, pred_sents, pred_scores,
                    match_scores_exact, match_scores_partial,
                    valid_pred_flags, present_pred_flags, pred_dict["copied_flags"])):
            # if p_id > 5:
            #     break

            preds_out += '%s\n' % (' '.join(word))
            if is_present:
                print_phrase = '[%s]' % ' '.join(word)
            else:
                print_phrase = ' '.join(word)

            if match == 1.0:
                correct_str = '[correct!]'
            else:
                correct_str = ''

            if any(copied_flag):
                copy_str = '[copied!]'
            else:
                copy_str = ''

            pred_str = '\t\t[%.4f]\t%s \t %s %s%s\n' % (-score, print_phrase, str(pred_idx), correct_str, copy_str)
            if not is_valid:
                pred_str = '\t%s' % pred_str

            print_out += pred_str

        # split tgts by present/absent
        present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
        absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if ~present]

        # filter out results of invalid preds
        valid_pred_sents = [seq for seq, valid in zip(pred_sents, valid_pred_flags) if valid]
        present_pred_flags = present_pred_flags[valid_pred_flags]

        match_scores_exact = match_scores_exact[valid_pred_flags]
        match_scores_partial = match_scores_partial[valid_pred_flags]
        match_scores_mixed = match_scores_mixed[valid_pred_flags]

        # split preds by present/absent and exact/partial/mixed
        present_preds = [pred for pred, present in zip(valid_pred_sents, present_pred_flags) if present]
        present_exact_match_scores = match_scores_exact[present_pred_flags]
        present_partial_match_scores = match_scores_partial[present_pred_flags][:, present_tgt_flags]
        present_mixed_match_scores = match_scores_mixed[present_pred_flags][:, present_tgt_flags]

        absent_preds = [pred for pred, present in zip(valid_pred_sents, present_pred_flags) if ~present]
        absent_exact_match_scores = match_scores_exact[~present_pred_flags]
        absent_partial_match_scores = match_scores_partial[~present_pred_flags][:, ~present_tgt_flags]
        absent_mixed_match_scores = match_scores_mixed[~present_pred_flags][:, ~present_tgt_flags]

        assert len(valid_pred_sents) == len(match_scores_exact) == len(present_pred_flags)
        assert len(present_preds) == len(present_exact_match_scores) == len(present_partial_match_scores) == len(present_mixed_match_scores)
        assert present_partial_match_scores.shape == present_mixed_match_scores.shape
        assert len(absent_preds) == len(absent_exact_match_scores) == len(absent_partial_match_scores) == len(absent_mixed_match_scores)
        assert absent_partial_match_scores.shape == absent_mixed_match_scores.shape

        # Compute metrics
        print_out += "\n ======================================================="
        # get the scores on different scores (for absent results, only recall matters)
        present_exact_results = run_metrics(present_exact_match_scores, present_preds, present_tgts, metric_names, topk_range)
        absent_exact_results = run_metrics(absent_exact_match_scores, absent_preds, absent_tgts, metric_names, absent_topk_range)
        present_partial_results = run_metrics(present_partial_match_scores, present_preds, present_tgts, metric_names, topk_range, type='partial')
        absent_partial_results = run_metrics(absent_partial_match_scores, absent_preds, absent_tgts, metric_names, absent_topk_range, type='partial')
        # present_mixed_results = run_metrics(present_mixed_match_scores, present_preds, present_tgts, metric_names, topk_range, type='partial')
        # absent_mixed_results = run_metrics(absent_mixed_match_scores, absent_preds, absent_tgts, metric_names, absent_topk_range, type='partial')

        def _gather_scores(gathered_scores, results_names, results_dicts):
            for result_name, result_dict in zip(results_names, results_dicts):
                for metric_name, score in result_dict.items():
                    if metric_name.endswith('_num'):
                        # if it's 'present_num' or 'absent_num', leave as is
                        field_name = result_name
                    else:
                        # if it's other score like 'precision@5' is renamed to like 'present_exact_precision@'
                        field_name = result_name+'_'+metric_name

                    if field_name not in gathered_scores:
                        gathered_scores[field_name] = []

                    gathered_scores[field_name].append(score)

            return gathered_scores

        results_names = ['present_exact', 'absent_exact',
                         'present_partial', 'absent_partial',
                         # 'present_mixed', 'absent_mixed'
                         ]
        results_list = [present_exact_results, absent_exact_results,
                        present_partial_results, absent_partial_results,
                        # present_mixed_results, absent_mixed_results
                        ]
        # update score_dict, appending new scores (results_list) to it
        score_dict = _gather_scores(score_dict, results_names, results_list)

        for name, resutls in zip(results_names, results_list):
            if name.startswith('present'):
                topk = 10
            else:
                topk = 50

            print_out += "\n --- batch {} P/R/F1 @{}: \t".format(name, topk) \
                         + " {:.4f} , {:.4f} , {:.4f}".format(resutls['precision@{}'.format(topk)],
                                                           resutls['recall@{}'.format(topk)],
                                                           resutls['f_score@{}'.format(topk)])
            print_out += "\n --- total {} P/R/F1 @{}: \t".format(name, topk) \
                         + " {:.4f} , {:.4f} , {:.4f}".format(np.average(score_dict['{}_precision@{}'.format(name, topk)]),
                                                           np.average(score_dict['{}_recall@{}'.format(name, topk)]),
                                                           np.average(score_dict['{}_f_score@{}'.format(name, topk)]))

        print_out += "\n ======================================================="

        if verbose:
            if logger:
                logger.info(print_out)
            else:
                print(print_out)

        # add phrase count for computing average performance on non-empty items
        results_names = ['present_num', 'absent_num']
        results_list = [{'present_num': len(present_tgts)}, {'absent_num': len(absent_tgts)}]
        score_dict = _gather_scores(score_dict, results_names, results_list)

    # for k, v in score_dict.items():
    #     print('%s, num=%d, mean=%f' % (k, len(v), np.average(v)))

    return score_dict


def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]


def macro_averaged_score(precisionlist, recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score = 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall), 2)
    return precision, recall, f_score


def get_match_result(true_seqs, pred_seqs, do_stem=True, type='exact'):
    '''
    If type='exact', returns a list of booleans indicating if a pred has a matching tgt
    If type='partial', returns a 2D matrix, each value v_ij is a float in range of [0,1]
        indicating the (jaccard) similarity between pred_i and tgt_j
    :param true_seqs:
    :param pred_seqs:
    :param do_stem:
    :param topn:
    :param type: 'exact' or 'partial'
    :return:
    '''
    # do processing to baseline predictions
    if type == "exact":
        match_score = np.zeros(shape=(len(pred_seqs)), dtype='float32')
    else:
        match_score = np.zeros(shape=(len(pred_seqs), len(true_seqs)), dtype='float32')

    target_number = len(true_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:
        true_seqs = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]

    for pred_id, pred_seq in enumerate(pred_seqs):
        if type == 'exact':
            match_score[pred_id] = 0
            for true_id, true_seq in enumerate(true_seqs):
                match = True
                if len(pred_seq) != len(true_seq):
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    match_score[pred_id] = 1
                    break
        elif type == 'ngram':
            # use jaccard coefficient as the similarity of partial match (1+2 grams)
            pred_seq_set = set(pred_seq)
            pred_seq_set.update(set([pred_seq[i]+'_'+pred_seq[i+1] for i in range(len(pred_seq)-1)]))
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                true_seq_set.update(set([true_seq[i]+'_'+true_seq[i+1] for i in range(len(true_seq)-1)]))
                similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                          / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                match_score[pred_id, true_id] = similarity
        elif type == 'mixed':
            # similar to jaccard, but addtional to 1+2 grams we also put in the full string, serves like a exact+partial surrogate
            pred_seq_set = set(pred_seq)
            pred_seq_set.update(set([pred_seq[i]+'_'+pred_seq[i+1] for i in range(len(pred_seq)-1)]))
            pred_seq_set.update(set(['_'.join(pred_seq)]))
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                true_seq_set.update(set([true_seq[i]+'_'+true_seq[i+1] for i in range(len(true_seq)-1)]))
                true_seq_set.update(set(['_'.join(true_seq)]))
                similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                          / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                match_score[pred_id, true_id] = similarity

        elif type == 'bleu':
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            # n-gras precision doesn't work that well
            for true_id, true_seq in enumerate(true_seqs):
                match_score[pred_id, true_id] = bleu(pred_seq, [true_seq], [0.7, 0.3, 0.0])

    return match_score


def run_metrics(match_list, pred_list, tgt_list, score_names, topk_range, type='exact'):
    """
    Return a dict of scores containing len(score_names) * len(topk_range) items
    score_names and topk_range actually only define the names of each score in score_dict.
    :param match_list:
    :param pred_list:
    :param tgt_list:
    :param score_names:
    :param topk_range:
    :return:
    """
    score_dict = {}
    if len(tgt_list) == 0:
        for score_name in score_names:
            for topk in topk_range:
                score_dict['{}@{}'.format(score_name, topk)] = 0.0
        return score_dict

    assert len(match_list) == len(pred_list)
    for topk in topk_range:
        if topk == 'k':
            cutoff = len(tgt_list)
        else:
            cutoff = topk

        if len(pred_list) > cutoff:
            pred_list_k = np.asarray(pred_list[:cutoff])
            match_list_k = match_list[:cutoff]
        else:
            pred_list_k = np.asarray(pred_list)
            match_list_k = match_list

        if type == 'partial':
            cost_matrix = np.asarray(match_list_k, dtype=float)
            # convert to a negative matrix because linear_sum_assignment() looks for minimal assignment
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cost_matrix)
            match_list_k = cost_matrix[row_ind, col_ind]
            overall_cost = cost_matrix[row_ind, col_ind].sum()
            '''
            print("\n%d" % topk)
            print(row_ind, col_ind)
            print("Pred" + str(np.asarray(pred_list)[row_ind].tolist()))
            print("Target" + str(tgt_list))
            print("Maximum Score: %f" % overall_cost)

            print("Pred list")
            for p_id, (pred, cost) in enumerate(zip(pred_list, cost_matrix)):
                print("\t%d \t %s - %s" % (p_id, pred, str(cost)))
            '''

        # Micro-Averaged Method
        micro_pk = float(sum(match_list_k)) / float(len(pred_list_k)) if len(pred_list_k) > 0 else 0.0
        micro_rk = float(sum(match_list_k)) / float(len(tgt_list)) if len(tgt_list) > 0 else 0.0

        if micro_pk + micro_rk > 0:
            micro_f1 = float(2 * (micro_pk * micro_rk)) / (micro_pk + micro_rk)
        else:
            micro_f1 = 0.0

        for score_name, v in zip(score_names, [micro_pk, micro_rk, micro_f1]):
            score_dict['{}@{}'.format(score_name, topk)] = v

    return score_dict


def f1_score(prediction, ground_truth):
    # both prediction and grount_truth should be list of words
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def self_redundancy(_input):
    # _input shoule be list of list of words
    if len(_input) == 0:
        return None
    _len = len(_input)
    scores = np.ones((_len, _len), dtype="float32") * -1.0
    for i in range(_len):
        for j in range(_len):
            if scores[i][j] != -1:
                continue
            elif i == j:
                scores[i][j] = 0.0
            else:
                f1 = f1_score(_input[i], _input[j])
                scores[i][j] = f1
                scores[j][i] = f1
    res = np.max(scores, 1)
    res = np.mean(res)
    return res


def init_opt():

    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--src', '-src', required=True,
                        help="Source file of groundtruth data.")
    parser.add_argument('--tgt', '-tgt', required=True,
                        help="Target file of groundtruth data.")
    parser.add_argument('--pred', '-pred', required=True,
                        help="File of predicted keyphrases, each line is a JSON dict.")
    parser.add_argument('--output_dir', '-output_dir',
                        help="Path to output log/results.")
    parser.add_argument('--unk_token', '-unk_token', default="<unk>",
                        help=".")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help=".")

    opt = parser.parse_args()

    return opt


def kp_results_to_str(results_dict):
    """
    return ">> ROUGE(1/2/3/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_su*_f_score"] * 100)
    """
    summary_dict = {}
    for k,v in results_dict.items():
        summary_dict[k] = np.average(v)

    return json.dumps(summary_dict)


def keyphrase_eval(src_path, tgt_path, pred_path, unk_token='<unk>', verbose=False, logger=None):
    src_data = [json.loads(l) for l in open(src_path, "r")]
    tgt_data = [json.loads(l) for l in open(tgt_path, "r")]
    preds = [json.loads(l) for l in open(pred_path, "r")]

    assert len(preds) == len(src_data) == len(tgt_data)

    results_dict = evaluate(src_data, tgt_data, preds, unk_token=unk_token, logger=logger, verbose=verbose)

    return results_dict


if __name__ == '__main__':
    opt = init_opt()
    logger = init_logger(opt.pred + ".eval.log")

    results_dict = keyphrase_eval(opt, logger)

    logger.info(kp_results_to_str(results_dict))
