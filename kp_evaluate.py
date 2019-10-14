
import argparse
import json

import scipy
import tqdm
from nltk.stem.porter import *
import numpy as np
from collections import Counter

import pandas as pd

import os

from nltk.translate.bleu_score import sentence_bleu as bleu

from onmt.keyphrase.utils import if_present_duplicate_phrases
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


def evaluate(src_list, tgt_list, pred_list, unk_token, logger=None, verbose=False, report_path=None, eval_topbeam=False):
    # progbar = Progbar(logger=logger, title='', target=len(pred_list), total_examples=len(pred_list))

    if report_path:
        report_file = open(report_path, 'w+')
    else:
        report_file = None
    # 'k' means the number of phrases in ground-truth
    topk_range = [5, 10, 'k', 'M']
    absent_topk_range = [10, 30, 50]
    # 'precision_hard' and 'f_score_hard' mean that precision is calculated with denominator strictly as K (say 5 or 10), won't be lessened even number of preds is smaller
    metric_names = ['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard']

    score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    '''
    process each example in current batch
    '''
    for i, (src_dict, tgt_dict, pred_dict) in tqdm.tqdm(enumerate(zip(src_list, tgt_list, pred_list))):
        src_seq = src_dict["src"].split()
        tgt_seqs =[t.split() for t in tgt_dict["tgt"]]

        if eval_topbeam:
            pred_sents = pred_dict["topseq_pred_sents"]
            pred_idxs = pred_dict["topseq_preds"] if "topseq_preds" in pred_dict else None
            pred_scores = pred_dict["topseq_pred_scores"] if "topseq_pred_scores" in pred_dict else None
            copied_flags = None
        else:
            pred_sents = pred_dict["pred_sents"]
            pred_idxs = pred_dict["preds"] if "preds" in pred_dict else None
            pred_scores = pred_dict["pred_scores"] if "pred_scores" in pred_dict else None
            copied_flags = pred_dict["copied_flags"] if "copied_flags" in pred_dict else None

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
        valid_pred_flags = valid_pred_flags * ~duplicate_flags if len(valid_pred_flags) > 0 else []
        valid_and_present = valid_pred_flags * present_pred_flags if len(valid_pred_flags) > 0 else []
        print_out += '\n[PREDICTION] #(valid)=%d, #(present)=%d, #(retained&present)=%d, #(all)=%d\n' % (sum(valid_pred_flags), sum(present_pred_flags), sum(valid_and_present), len(pred_sents))
        print_out += ''

        # compute match scores (exact, partial and mixed), for exact it's a list otherwise matrix
        match_scores_exact = get_match_result(true_seqs=tgt_seqs, pred_seqs=pred_sents, type='exact')
        match_scores_partial = get_match_result(true_seqs=tgt_seqs, pred_seqs=pred_sents, type='ngram')
        # simply add full-text to n-grams might not be good as its contribution is not clear
        match_scores_mixed = get_match_result(true_seqs=tgt_seqs, pred_seqs=pred_sents, type='mixed')

        # sanity check of pred (does not work for eval_topbeam, discard)
        # num_pred = len(pred_dict["pred_sents"])
        # for d_name, d in zip(['pred_idxs', 'pred_sents', 'pred_scores',
        #               'match_scores_exact', 'valid_pred_flags',
        #               'present_pred_flags', 'copied_flags'],
        #               [pred_idxs, pred_sents, pred_scores,
        #               match_scores_exact, valid_pred_flags,
        #               present_pred_flags, copied_flags]):
        #     if d is not None:
        #         if len(d) != num_pred:
        #             logger.error('%s number does not match' % d_name)
        #         assert len(d) == num_pred

        '''
        Print and export predictions
        '''
        preds_out = ''
        for p_id, (word, match, match_soft,
                   is_valid, is_present) in enumerate(
                zip(pred_sents, match_scores_exact, match_scores_partial,
                    valid_pred_flags, present_pred_flags)):
            # if p_id > 5:
            #     break
            score = pred_scores[p_id] if pred_scores else "Score N/A"
            pred_idx = pred_idxs[p_id] if pred_idxs else "Index N/A"
            copied_flag = copied_flags[p_id] if copied_flags else "CopyFlag N/A"

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

            pred_str = '\t\t%s\t%s \t %s %s%s\n' % ('[%.4f]' % (-score) if pred_scores else "Score N/A",
                                                    print_phrase, str(pred_idx),
                                                    correct_str, copy_str)
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
        absent_preds = [pred for pred, present in zip(valid_pred_sents, present_pred_flags) if ~present]
        if len(present_pred_flags) > 0:
            present_exact_match_scores = match_scores_exact[present_pred_flags]
            present_partial_match_scores = match_scores_partial[present_pred_flags][:, present_tgt_flags]
            present_mixed_match_scores = match_scores_mixed[present_pred_flags][:, present_tgt_flags]
            absent_exact_match_scores = match_scores_exact[~present_pred_flags]
            absent_partial_match_scores = match_scores_partial[~present_pred_flags][:, ~present_tgt_flags]
            absent_mixed_match_scores = match_scores_mixed[~present_pred_flags][:, ~present_tgt_flags]
        else:
            present_exact_match_scores = []
            present_partial_match_scores = []
            present_mixed_match_scores = []
            absent_exact_match_scores = []
            absent_partial_match_scores = []
            absent_mixed_match_scores = []

        # assert len(valid_pred_sents) == len(match_scores_exact) == len(present_pred_flags)
        # assert len(present_preds) == len(present_exact_match_scores) == len(present_partial_match_scores) == len(present_mixed_match_scores)
        # assert present_partial_match_scores.shape == present_mixed_match_scores.shape
        # assert len(absent_preds) == len(absent_exact_match_scores) == len(absent_partial_match_scores) == len(absent_mixed_match_scores)
        # assert absent_partial_match_scores.shape == absent_mixed_match_scores.shape

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
                        # if it's 'present_tgt_num' or 'absent_tgt_num', leave as is
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
                topk = 5
            else:
                topk = 50

            print_out += "\n --- batch {} P/R/F1/Corr @{}: \t".format(name, topk) \
                         + " {:.4f} , {:.4f} , {:.4f} , {:2f}".format(resutls['precision@{}'.format(topk)],
                                                           resutls['recall@{}'.format(topk)],
                                                           resutls['f_score@{}'.format(topk)],
                                                           resutls['correct@{}'.format(topk)])
            print_out += "\n --- total {} P/R/F1/Corr @{}: \t".format(name, topk) \
                         + " {:.4f} , {:.4f} , {:.4f} , {:2f}".format(np.average(score_dict['{}_precision@{}'.format(name, topk)]),
                                                           np.average(score_dict['{}_recall@{}'.format(name, topk)]),
                                                           np.average(score_dict['{}_f_score@{}'.format(name, topk)]),
                                                           np.sum(score_dict['{}_correct@{}'.format(name, topk)]))

            if name.startswith('present'):
                topk = 10
                print_out += "\n --- batch {} P/R/F1/Corr @{}: \t".format(name, topk) \
                             + " {:.4f} , {:.4f} , {:.4f} , {:2f}".format(resutls['precision@{}'.format(topk)],
                                                               resutls['recall@{}'.format(topk)],
                                                               resutls['f_score@{}'.format(topk)],
                                                               resutls['correct@{}'.format(topk)])
                print_out += "\n --- total {} P/R/F1/Corr @{}: \t".format(name, topk) \
                             + " {:.4f} , {:.4f} , {:.4f} , {:2f}".format(np.average(score_dict['{}_precision@{}'.format(name, topk)]),
                                                               np.average(score_dict['{}_recall@{}'.format(name, topk)]),
                                                               np.average(score_dict['{}_f_score@{}'.format(name, topk)]),
                                                               np.sum(score_dict['{}_correct@{}'.format(name, topk)]))


        print_out += "\n ======================================================="

        if verbose:
            if logger:
                logger.info(print_out)
            else:
                print(print_out)

        if report_file:
            report_file.write(print_out)

        # add tgt/pred count for computing average performance on non-empty items
        results_names = ['present_tgt_num', 'absent_tgt_num', 'present_pred_num', 'absent_pred_num', 'unique_pred_num', 'dup_pred_num', 'beam_num', 'beamstep_num']
        results_list = [{'present_tgt_num': len(present_tgts)},
                        {'absent_tgt_num': len(absent_tgts)},
                        {'present_pred_num': len(present_preds)},
                        {'absent_pred_num': len(absent_preds)},
                        {'unique_pred_num': pred_dict['unique_pred_num'] if 'unique_pred_num' in pred_dict else 0},
                        {'dup_pred_num': pred_dict['dup_pred_num'] if 'dup_pred_num' in pred_dict else 0},
                        {'beam_num': pred_dict['beam_num'] if 'beam_num' in pred_dict else 0},
                        {'beamstep_num': pred_dict['beamstep_num'] if 'beamstep_num' in pred_dict else 0},
                        ]
        score_dict = _gather_scores(score_dict, results_names, results_list)

    # for k, v in score_dict.items():
    #     print('%s, num=%d, mean=%f' % (k, len(v), np.average(v)))

    if report_file:
        report_file.close()

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
                if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
                    similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                              / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                else:
                    similarity = 0.0
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
                if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
                    similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                              / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                else:
                    similarity = 0.0
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
        for topk in topk_range:
            for score_name in score_names:
                score_dict['{}@{}'.format(score_name, topk)] = 0.0
        return score_dict

    assert len(match_list) == len(pred_list)
    for topk in topk_range:
        if topk == 'k':
            cutoff = len(tgt_list)
        elif topk == 'M':
            cutoff = len(pred_list)
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
            if len(match_list_k) > 0:
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
        correct_num = int(sum(match_list_k))
        # Precision, Recall and F-score, with flexible cutoff (if number of pred is smaller)
        micro_p = float(sum(match_list_k)) / float(len(pred_list_k)) if len(pred_list_k) > 0 else 0.0
        micro_r = float(sum(match_list_k)) / float(len(tgt_list)) if len(tgt_list) > 0 else 0.0

        if micro_p + micro_r > 0:
            micro_f1 = float(2 * (micro_p * micro_r)) / (micro_p + micro_r)
        else:
            micro_f1 = 0.0
        # F-score, with a hard cutoff on precision, offset the favor towards fewer preds
        micro_p_hard = float(sum(match_list_k)) / cutoff if len(pred_list_k) > 0 else 0.0
        if micro_p_hard + micro_r > 0:
            micro_f1_hard = float(2 * (micro_p_hard * micro_r)) / (micro_p_hard + micro_r)
        else:
            micro_f1_hard = 0.0

        for score_name, v in zip(score_names, [correct_num, micro_p, micro_r, micro_f1, micro_p_hard, micro_f1_hard]):
            score_dict['{}@{}'.format(score_name, topk)] = v

    return score_dict


def f1_score(prediction, ground_truth):
    # both prediction and grount_truth should be list of words
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction) if len(prediction) > 0 else 0.0
    recall = 1.0 * num_same / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if len(precision + recall) > 0 else 0.0
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


def baseline_pred_loader(pred_path, model_name):
    pred_dict_list = []

    if model_name in ['tfidf', 'textrank', 'singlerank', 'expandrank', 'maui', 'kea']:
        doc_list = [file_name for file_name in os.listdir(pred_path) if file_name.endswith('txt.phrases')]
        doc_list = sorted(doc_list, key=lambda k: int(k[:k.index('.txt.phrases')]))
        for doc_name in doc_list:
            doc_path = os.path.join(pred_path, doc_name)
            pred_dict = {}
            pred_dict['pred_sents'] = []

            for l in open(doc_path, 'r').readlines():
                pred_dict['pred_sents'].append(l.lower().split())
            pred_dict_list.append(pred_dict)
    else:
        raise NotImplementedError

    return pred_dict_list


def keyphrase_eval(src_path, tgt_path, pred_path, unk_token='<unk>', verbose=False, logger=None, report_path=None, eval_topbeam=False, model_name='nn'):
    src_data = [json.loads(l) for l in open(src_path, "r")]
    tgt_data = [json.loads(l) for l in open(tgt_path, "r")]
    if model_name == 'nn':
        pred_data = [json.loads(l) for l in open(pred_path, "r")]
    else:
        pred_data = baseline_pred_loader(pred_path, model_name)

    if len(pred_data) == len(src_data) == len(tgt_data):
        results_dict = evaluate(src_data, tgt_data, pred_data, unk_token=unk_token, logger=logger, verbose=verbose, report_path=report_path, eval_topbeam=eval_topbeam)
        return results_dict
    else:
        logger.info("#(src)=%d, #(tgt)=%d, #(pred)=%d" % (len(src_data), len(tgt_data), len(pred_data)))
        return None




def summarize_scores(ckpt_name, score_dict):
    avg_dict = {}
    avg_dict['checkpoint_name'] = ckpt_name
    if ckpt_name.find('_') > 0:
        avg_dict['model_name'] = '_'.join(ckpt_name.rsplit('_')[:-1])
        avg_dict['#train_step'] = ckpt_name.rsplit('_')[-1]
    else:
        avg_dict['model_name'] = ckpt_name
        avg_dict['#train_step'] = ''

    # doc stat
    avg_dict['#doc'] = len(score_dict['present_tgt_num'])
    avg_dict['#pre_doc'] = len([x for x in score_dict['present_tgt_num'] if x > 0])
    avg_dict['#ab_doc'] = len([x for x in score_dict['absent_tgt_num'] if x > 0])

    # tgt & pred stat
    if 'present_tgt_num' in score_dict and 'absent_tgt_num' in score_dict:
        avg_dict['#tgt'] = sum(score_dict['present_tgt_num']) + sum(score_dict['absent_tgt_num'])
        avg_dict['#pre_tgt'] = sum(score_dict['present_tgt_num'])
        avg_dict['#ab_tgt'] = sum(score_dict['absent_tgt_num'])
    else:
        avg_dict['#tgt'] = 0
        avg_dict['#pre_tgt'] = 0
        avg_dict['#ab_tgt'] = 0

    if 'present_pred_num' in score_dict and 'absent_pred_num' in score_dict:
        avg_dict['#pred'] = sum(score_dict['present_pred_num']) + sum(score_dict['absent_pred_num'])
        avg_dict['#pre_pred'] = sum(score_dict['present_pred_num'])
        avg_dict['#ab_pred'] = sum(score_dict['absent_pred_num'])
    else:
        avg_dict['#pred'] = 0
        avg_dict['#pre_pred'] = 0
        avg_dict['#ab_pred'] = 0

    avg_dict['#unique_pred'] = sum(score_dict['unique_pred_num']) if 'unique_pred_num' in score_dict else 0
    avg_dict['#dup_pred'] = sum(score_dict['dup_pred_num']) if 'dup_pred_num' in score_dict else 0
    avg_dict['#beam'] = sum(score_dict['beam_num']) if 'beam_num' in score_dict else 0
    avg_dict['#beamstep'] = sum(score_dict['beamstep_num']) if 'beamstep_num' in score_dict else 0

    present_tgt_num = score_dict['present_tgt_num'] if 'present_tgt_num' in score_dict else 0
    absent_tgt_num = score_dict['absent_tgt_num'] if 'absent_tgt_num' in score_dict else 0

    if 'unique_pred_num' in score_dict: del score_dict['present_tgt_num']
    if 'absent_tgt_num' in score_dict: del score_dict['absent_tgt_num']
    if 'present_pred_num' in score_dict: del score_dict['present_pred_num']
    if 'absent_pred_num' in score_dict: del score_dict['absent_pred_num']
    if 'unique_pred_num' in score_dict: del score_dict['unique_pred_num']
    if 'dup_pred_num' in score_dict: del score_dict['dup_pred_num']
    if 'beam_num' in score_dict: del score_dict['beam_num']
    if 'beamstep_num' in score_dict: del score_dict['beamstep_num']

    for score_name, score_list in score_dict.items():
        # number of correct phrases
        if score_name.find('correct') > 0:
            # only keep exact results (partial count is trivial)
            if score_name.find('exact') > 0:
                avg_dict[score_name] = np.sum(score_list)
            continue

        # various scores (precision, recall, f-score)
        if score_name.startswith('present'):
            tmp_scores = [score for score, num in zip(score_list, present_tgt_num) if num > 0]
            avg_dict[score_name] = np.average(tmp_scores)
        elif score_name.startswith('absent'):
            tmp_scores = [score for score, num in zip(score_list, absent_tgt_num) if num > 0]
            avg_dict[score_name] = np.average(tmp_scores)
        else:
            logger.error("NotImplementedError: found key %s" % score_name)
            raise NotImplementedError

    columns = list(avg_dict.keys())
    # print(columns)
    summary_df = pd.DataFrame.from_dict(avg_dict, orient='index').transpose()[columns]
    # print('\n')
    # print(list(summary_df.columns))
    # input()

    return summary_df


def export_summary_to_csv(json_root_dir, report_csv_path):
    dataset_scores_dict = {}

    # total_file_num = len(*[[file for file in files if file.endswith('.json')] for subdir, dirs, files in os.walk(json_root_dir)])
    # file_count = 0

    for subdir, dirs, files in os.walk(json_root_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            # file_count += 1
            # print("file_count/file_num=%d/%d" % (file_count, total_file_num))

            file_name = file[: file.find('.json')]
            ckpt_name = file_name[: file.rfind('-')] if file.find('-') > 0 else file_name
            dataset_name = file_name[file.rfind('-')+1: ]
            # if dataset_name != "kp20k_valid2k":
            #     print("Skip "+dataset_name)
            #     continue
            # key is dataset name, value is a dict whose key is metric name and value is a list of floats
            score_dict = json.load(open(os.path.join(subdir, file), 'r'))
            # ignore scores where no tgts available and return the average
            score_df = summarize_scores(ckpt_name, score_dict)

            if dataset_name in dataset_scores_dict:
                dataset_scores_dict[dataset_name] = dataset_scores_dict[dataset_name].append(score_df)
            else:
                dataset_scores_dict[dataset_name] = score_df

    for dataset, score_df in dataset_scores_dict.items():
        # if dataset_name != "kp20k_valid2k":
        #     continue
        print("Writing summary to: %s" % report_csv_path % dataset)
        score_df.to_csv(report_csv_path % dataset)

def init_opt():

    parser = argparse.ArgumentParser()
    # Input/output options
    parser.add_argument('--data', '-data', required=True,
                        help="Path to the source/target file of groundtruth data.")
    parser.add_argument('--pred_dir', '-pred_dir', required=True,
                        help="Directory to pred folders, each folder contains .pred files, each line is a JSON dict about predicted keyphrases.")
    parser.add_argument('--output_dir', '-output_dir',
                        help="Path to output log/results.")
    parser.add_argument('--unk_token', '-unk_token', default="<unk>",
                        help=".")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help=".")
    parser.add_argument('--eval_topbeam', '-eval_topbeam', action='store_true', required=False, help='(only useful for one2seq models) Evaluate with all sequences or just take the top-score sequence.')

    parser.add_argument('-testsets', nargs='+', type=str, default=["inspec", "krapivin", "nus", "semeval", "duc"], help='Specify datasets to test on')
    # parser.add_argument('-testsets', nargs='+', type=str, default=["duc", "inspec", "krapivin", "nus", "semeval"], help='Specify datasets to test on')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = init_opt()
    score_dicts = {}

    for ckpt_name in os.listdir(opt.pred_dir):

        for dataname in opt.testsets:
            src_path = os.path.join(opt.data, dataname, "%s_test.src" % dataname)
            tgt_path = os.path.join(opt.data, dataname, "%s_test.tgt" % dataname)
            pred_path = os.path.join(opt.pred_dir, ckpt_name, "%s.pred" % dataname)

            if not os.path.exists(opt.output_dir):
                os.makedirs(opt.output_dir)
            if not os.path.exists(os.path.join(opt.output_dir, 'pred', ckpt_name)):
                os.makedirs(os.path.join(opt.output_dir, 'pred', ckpt_name))
            if not os.path.exists(os.path.join(opt.output_dir, 'eval')):
                os.makedirs(os.path.join(opt.output_dir, 'eval'))

            logger = init_logger(opt.output_dir + "kp_evaluate.%s.eval.log" % dataname)
            report_path = os.path.join(opt.output_dir, 'pred', ckpt_name, '%s.report.txt' % dataname)
            score_path = os.path.join(opt.output_dir, 'eval', ckpt_name + '-%s.json' % dataname)

            logger.info("Evaluating %s" % dataname)

            if not os.path.exists(score_path):
                score_dict = keyphrase_eval(src_path=src_path,
                                              tgt_path=tgt_path,
                                              pred_path=pred_path,
                                              unk_token = '<unk>',
                                              verbose = opt.verbose,
                                              logger = logger,
                                              report_path = report_path,
                                              eval_topbeam=opt.eval_topbeam
                                            )
                logger.info(kp_results_to_str(score_dict))

                with open(score_path, 'w') as output_json:
                    output_json.write(json.dumps(score_dict))

                score_dicts[dataname] = score_dict

        export_summary_to_csv(json_root_dir=os.path.join(opt.output_dir, 'eval'),
                              report_csv_path=os.path.join(opt.output_dir, 'summary_%s.csv' % ('%s')))

        logger.info("Done!")