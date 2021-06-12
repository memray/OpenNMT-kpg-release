# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import math
from collections import Counter

import scipy
import numpy as np
from sklearn import metrics

from onmt.keyphrase.utils import stem_word_list

from nltk.translate.bleu_score import sentence_bleu as bleu

from nltk.stem.porter import *
stemmer = PorterStemmer()

import spacy
spacy_nlp = spacy.load('en_core_web_sm')
from onmt.keyphrase.utils import validate_phrases, if_present_duplicate_phrases

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"



def compute_match_scores(tgt_seqs, pred_seqs, do_lower=True, do_stem=True, type='exact'):
    '''
    If type='exact', returns a list of booleans indicating if a pred has a matching tgt
    If type='partial', returns a 2D matrix, each value v_ij is a float in range of [0,1]
        indicating the (jaccard) similarity between pred_i and tgt_j
    :param tgt_seqs:
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
        match_score = np.zeros(shape=(len(pred_seqs), len(tgt_seqs)), dtype='float32')

    target_number = len(tgt_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_lower:
        tgt_seqs = [[w.lower() for w in seq] for seq in tgt_seqs]
        pred_seqs = [[w.lower() for w in seq] for seq in pred_seqs]
    if do_stem:
        tgt_seqs = [stem_word_list(seq) for seq in tgt_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]

    for pred_id, pred_seq in enumerate(pred_seqs):
        if type == 'exact':
            match_score[pred_id] = 0
            for true_id, true_seq in enumerate(tgt_seqs):
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
            for true_id, true_seq in enumerate(tgt_seqs):
                true_seq_set = set(true_seq)
                true_seq_set.update(set([true_seq[i]+'_'+true_seq[i+1] for i in range(len(true_seq)-1)]))
                if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
                    similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                              / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                else:
                    similarity = 0.0
                match_score[pred_id, true_id] = similarity
        elif type == 'mixed':
            # similar to jaccard, but addtional to 1+2 grams we also put in the full string, serves like an exact+partial surrogate
            pred_seq_set = set(pred_seq)
            pred_seq_set.update(set([pred_seq[i]+'_'+pred_seq[i+1] for i in range(len(pred_seq)-1)]))
            pred_seq_set.update(set(['_'.join(pred_seq)]))
            for true_id, true_seq in enumerate(tgt_seqs):
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
            # n-grams precision doesn't work that well
            for true_id, true_seq in enumerate(tgt_seqs):
                match_score[pred_id, true_id] = bleu(pred_seq, [true_seq], [0.7, 0.3, 0.0])

    return match_score


def run_classic_metrics(match_list, pred_list, tgt_list, score_names, topk_range, type='exact'):
    """
    Return a dict of scores containing len(score_names) * len(topk_range) items
    score_names and topk_range actually only define the names of each score in score_dict.
    :param match_list:
    :param pred_list:
    :param tgt_list:
    :param score_names:
    :param topk_range:
    :param type: exact or partial
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

        for score_name, v in zip(['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard'], [correct_num, micro_p, micro_r, micro_f1, micro_p_hard, micro_f1_hard]):
            score_dict['{}@{}'.format(score_name, topk)] = v

    # return only the specified scores
    return_scores = {}
    for topk in topk_range:
        for score_name in score_names:
            return_scores['{}@{}'.format(score_name, topk)] = score_dict['{}@{}'.format(score_name, topk)]

    return return_scores


def run_advanced_metrics(match_scores, pred_list, tgt_list):
    score_dict = {}
    corrects, precisions, recalls, fscores = compute_PRF1(match_scores, pred_list, tgt_list)
    auc = compute_PR_AUC(precisions, recalls)
    ap = compute_AP(match_scores, precisions, tgt_list)
    mrr = compute_MRR(match_scores)
    sadr = compute_SizeAdjustedDiscountedRecall(match_scores, tgt_list)
    ndcg = compute_NormalizedDiscountedCumulativeGain(match_scores, tgt_list)
    alpha_ndcg_5 = compute_alphaNormalizedDiscountedCumulativeGain(pred_list, tgt_list, k=5, alpha=0.5)
    alpha_ndcg_10 = compute_alphaNormalizedDiscountedCumulativeGain(pred_list, tgt_list, k=10, alpha=0.5)

    score_dict['auc'] = auc
    score_dict['ap'] = ap
    score_dict['mrr'] = mrr
    score_dict['sadr'] = sadr
    score_dict['ndcg'] = ndcg
    score_dict['alpha_ndcg@5'] = alpha_ndcg_5
    score_dict['alpha_ndcg@10'] = alpha_ndcg_10

    # print('\nMatch[#=%d]=%s' % (len(match_scores), str(match_scores)))
    # print('Accum Corrects=' + str(corrects))
    # print('P@x=' + str(precisions))
    # print('R@x=' + str(recalls))
    # print('F-score@x=' + str(fscores))
    #
    # print('F-score@5=%f' % fscores[4])
    # print('F-score@10=%f' % (fscores[9] if len(fscores) > 9 else -9999))
    # print('F-score@O=%f' % fscores[len(tgt_list) - 1])
    # print('F-score@M=%f' % fscores[len(match_scores) - 1])
    #
    # print('AUC=%f' % auc)
    # print('AP=%f' % ap)
    # print('MRR=%f' % mrr)
    # print('SADR=%f' % sadr)
    # print('nDCG=%f' % ndcg)
    # print('α-nDCG@5=%f' % alpha_ndcg_5)
    # print('α-nDCG@10=%f' % alpha_ndcg_10)

    return score_dict


def compute_PRF1(match_scores, preds, tgts):
    corrects, precisions, recalls, fscores = [], [], [], []

    for pred_id, score in enumerate(match_scores):
        _corr = corrects[-1] + score if len(corrects) > 0 else score
        _p = _corr / (pred_id + 1) if pred_id + 1 > 0 else 0.0
        _r = _corr / len(tgts) if len(tgts) > 0 else 0.0
        _f1 = float(2 * (_p * _r)) / (_p + _r) if (_p + _r) > 0 else 0.0
        corrects += [_corr]
        precisions += [_p]
        recalls += [_r]
        fscores += [_f1]

    return corrects, precisions, recalls, fscores


def compute_MRR(match_scores):
    # A modified mean reciprocal rank for KP eval
    # MRR in IR uses the rank of first correct result. We use the rank of all correctly recalled results.
    # But it doesn't consider the missing predictions, so it's a precision-like metric
    mrr = 0.0
    count = 0.0

    for idx, match_score in enumerate(match_scores):
        if match_score == 0.0:
            continue
        mrr += match_score / (idx + 1)
        count += 1.0

    if count > 0:
        mrr /= count
    else:
        mrr = 0.0
    return mrr


def compute_AP(match_scores, precisions, tgts):
    # Average Precision: Note that the average is over all relevant documents and the relevant documents not retrieved get a precision score of zero.
    # Updated on March 4, 2020. Previously we average over the number of correct predictions.
    ap = 0.0
    tgt_count = len(tgts)

    for idx, (match_score, precision) in enumerate(zip(match_scores, precisions)):
        if match_score == 0.0:
            continue
        ap += precision

    if tgt_count > 0:
        ap /= tgt_count
    else:
        ap = 0.0
    return ap


def compute_PR_AUC(precisions, recalls):
    # we need to pad two values as the begin/end point of the curve
    p = [1.0] + precisions + [0.0]
    r = [0.0] + recalls + [(recalls[-1] if len(recalls) > 0 else 0.0)]
    pr_auc = metrics.auc(r, p)

    return pr_auc


def compute_SizeAdjustedDiscountedRecall(match_scores, tgts):
    # add a log2(pos-num_tgt+2) discount to correct predictions out of the top-k list
    cumulated_gain = 0.0
    num_tgts = len(tgts)

    for idx, match_score in enumerate(match_scores):
        if match_score == 0.0:
            continue
        if idx + 1 > num_tgts:
            gain = 1.0 / math.log(idx - num_tgts + 3, 2)
        else:
            gain = 1.0
        # print('gain@%d=%f' % (idx + 1, gain))
        cumulated_gain += gain

    if num_tgts > 0:
        ndr = cumulated_gain / num_tgts
    else:
        ndr = 0.0

    return ndr


def compute_NormalizedDiscountedCumulativeGain(match_scores, tgts):
    # add a positional discount to all predictions
    def _compute_dcg(match_scores):
        cumulated_gain = 0.0
        for idx, match_score in enumerate(match_scores):
            gain = match_score / math.log(idx + 2, 2)
            #             print('gain@%d=%f' % (idx + 1, gain))
            cumulated_gain += gain
        return cumulated_gain

    num_tgts = len(tgts)
    assert sum(match_scores) <= num_tgts, "Sum of relevance scores shouldn't exceed number of targets."
    if num_tgts > 0:
        dcg = _compute_dcg(match_scores)
        #         print('DCG=%f' % dcg)
        idcg = _compute_dcg([1.0] * num_tgts)
        #         print('IDCG=%f' % idcg)
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    #     print('nDCG=%f' % ndcg)
    return ndcg


def compute_alphaNormalizedDiscountedCumulativeGain(preds, tgts, k=5, alpha=0.5):
    # α-nDCG@k
    # add a positional discount to all predictions, and penalize repetive predictions
    def _compute_dcg(match_scores, novelty_scores, alpha):
        cumulated_gain = 0.0
        for idx, (match_score, novelty_score) in enumerate(zip(match_scores, novelty_scores)):
            gain = match_score * ((1 - alpha) ** (novelty_score)) / math.log(idx + 2, 2)
            # print('gain@%d=%f' % (idx + 1, gain))
            cumulated_gain += gain
        return cumulated_gain

    def _compute_matching_novelty_scores(preds, tgts):
        preds = [set(stem_word_list(seq)) for seq in preds]
        tgts = [set(stem_word_list(seq)) for seq in tgts]
        match_scores = [0.0] * len(preds)
        novelty_discounts = [0.0] * len(preds)
        rel_matrix = np.asarray([[0.0] * len(preds)] * len(tgts))

        for pred_id, pred in enumerate(preds):
            match_score = 0.0
            novelty_discount = 0.0
            for tgt_id, tgt in enumerate(tgts):
                if tgt.issubset(pred) or pred.issubset(tgt):
                    rel_matrix[tgt_id][pred_id] = 1.0
                    match_score = 1.0
                    if pred_id > 0 and sum(rel_matrix[tgt_id][: pred_id]) > novelty_discount:
                        novelty_discount = sum(rel_matrix[tgt_id][: pred_id])
            match_scores[pred_id] = match_score
            novelty_discounts[pred_id] = novelty_discount

        #         print('PRED[%d]=%s' % (len(preds), str(preds)))
        #         print('GT[%d]=%s' % (len(tgts), str(tgts)))
        #         print(match_scores)
        #         print(novelty_discounts)
        #         print(np.asarray(rel_matrix))
        return match_scores, novelty_discounts

    num_tgts = len(tgts)
    k = min(k, num_tgts)
    preds = preds[: k] if len(preds) > k else preds

    if num_tgts > 0:
        match_scores, novelty_discounts = _compute_matching_novelty_scores(preds, tgts)
        dcg = _compute_dcg(match_scores, novelty_discounts, alpha=alpha)
        idcg = _compute_dcg([1.0] * num_tgts, [0.0] * num_tgts, alpha=alpha)
        ndcg = dcg / idcg
    else:
        ndcg, dcg, idcg = 0.0, 0.0, 0.0

    # print('DCG=%f' % dcg)
    # print('IDCG=%f' % idcg)
    # print('nDCG=%f' % ndcg)
    return ndcg


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


def macro_averaged_score(precisionlist, recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score = 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall), 2)
    return precision, recall, f_score


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


def eval_and_print(src_text, tgt_kps, pred_kps, pred_scores, unk_token='<unk>'):
    src_seq = [t for t in re.split(r'\W', src_text) if len(t) > 0]
    tgt_seqs = [[t for t in re.split(r'\W', p) if len(t) > 0] for p in tgt_kps]
    pred_seqs = [[t for t in re.split(r'\W', p) if len(t) > 0] for p in pred_kps]

    topk_range = ['k', 10]
    absent_topk_range = ['M']
    metric_names = ['f_score']

    # 1st filtering, ignore phrases having <unk> and puncs
    valid_pred_flags = validate_phrases(pred_seqs, unk_token)
    # 2nd filtering: filter out phrases that don't appear in text, and keep unique ones after stemming
    present_pred_flags, _, duplicate_flags = if_present_duplicate_phrases(src_seq, pred_seqs)
    # treat duplicates as invalid
    valid_pred_flags = valid_pred_flags * ~duplicate_flags if len(valid_pred_flags) > 0 else []
    valid_and_present_flags = valid_pred_flags * present_pred_flags if len(valid_pred_flags) > 0 else []
    valid_and_absent_flags = valid_pred_flags * ~present_pred_flags if len(valid_pred_flags) > 0 else []

    # compute match scores (exact, partial and mixed), for exact it's a list otherwise matrix
    match_scores_exact = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs,
                                              do_lower=True, do_stem=True, type='exact')
    # split tgts by present/absent
    present_tgt_flags, _, _ = if_present_duplicate_phrases(src_seq, tgt_seqs)
    present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
    absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if ~present]

    # filter out results of invalid preds
    valid_preds = [seq for seq, valid in zip(pred_seqs, valid_pred_flags) if valid]
    valid_present_pred_flags = present_pred_flags[valid_pred_flags]

    valid_match_scores_exact = match_scores_exact[valid_pred_flags]

    # split preds by present/absent and exact/partial/mixed
    valid_present_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if present]
    valid_absent_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if ~present]
    present_exact_match_scores = valid_match_scores_exact[valid_present_pred_flags]
    absent_exact_match_scores = valid_match_scores_exact[~valid_present_pred_flags]

    all_exact_results = run_classic_metrics(valid_match_scores_exact, valid_preds, tgt_seqs, metric_names, topk_range)
    present_exact_results = run_classic_metrics(present_exact_match_scores, valid_present_preds, present_tgts, metric_names, topk_range)
    absent_exact_results = run_classic_metrics(absent_exact_match_scores, valid_absent_preds, absent_tgts, metric_names, absent_topk_range)

    eval_results_names = ['all_exact', 'present_exact', 'absent_exact']
    eval_results_list = [all_exact_results, present_exact_results, absent_exact_results]

    print_out = print_predeval_result(src_text,
                                      tgt_seqs, present_tgt_flags,
                                      pred_seqs, pred_scores, present_pred_flags, valid_pred_flags,
                                      valid_and_present_flags, valid_and_absent_flags, match_scores_exact,
                                      eval_results_names, eval_results_list)
    return print_out


def print_predeval_result(src_text,
                          tgt_seqs, present_tgt_flags,
                          pred_seqs, pred_scores, present_pred_flags, valid_pred_flags,
                          valid_and_present_flags, valid_and_absent_flags, match_scores_exact,
                          results_names, results_list):
    print_out = '=' * 50
    print_out += '\n[Source]: %s \n' % (src_text)

    print_out += '[GROUND-TRUTH] #(all)=%d, #(present)=%d, #(absent)=%d\n' % \
                 (len(present_tgt_flags), sum(present_tgt_flags), len(present_tgt_flags)-sum(present_tgt_flags))
    print_out += '\n'.join(
        ['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in
         zip(tgt_seqs, present_tgt_flags)])

    print_out += '\n[PREDICTION] #(all)=%d, #(valid)=%d, #(present)=%d, ' \
                 '#(valid&present)=%d, #(valid&absent)=%d\n' % (
        len(pred_seqs), sum(valid_pred_flags), sum(present_pred_flags),
        sum(valid_and_present_flags), sum(valid_and_absent_flags))
    print_out += ''
    preds_out = ''
    for p_id, (word, match, is_valid, is_present) in enumerate(
        zip(pred_seqs, match_scores_exact, valid_pred_flags, present_pred_flags)):
        score = pred_scores[p_id] if pred_scores else "Score N/A"

        preds_out += '%s\n' % (' '.join(word))
        if is_present:
            print_phrase = '[%s]' % ' '.join(word)
        else:
            print_phrase = ' '.join(word)

        if match == 1.0:
            correct_str = '[correct!]'
        else:
            correct_str = ''

        pred_str = '\t\t%s\t%s \t%s\n' % ('[%.4f]' % (-score) if pred_scores else "Score N/A",
                                                print_phrase, correct_str)
        if not is_valid:
            pred_str = '\t%s' % pred_str

        print_out += pred_str

    print_out += "\n ======================================================= \n"

    print_out += '[GROUND-TRUTH] #(all)=%d, #(present)=%d, #(absent)=%d\n' % \
                 (len(present_tgt_flags), sum(present_tgt_flags), len(present_tgt_flags)-sum(present_tgt_flags))
    print_out += '\n[PREDICTION] #(all)=%d, #(valid)=%d, #(present)=%d, ' \
                 '#(valid&present)=%d, #(valid&absent)=%d\n' % (
        len(pred_seqs), sum(valid_pred_flags), sum(present_pred_flags),
        sum(valid_and_present_flags), sum(valid_and_absent_flags))

    for name, results in zip(results_names, results_list):
        # print @5@10@O@M for present_exact, print @50@M for absent_exact
        if name in ['all_exact', 'present_exact', 'absent_exact']:
            if name.startswith('all') or name.startswith('present'):
                topk_list = ['10', 'k']
            else:
                topk_list = ['M']

            for topk in topk_list:
                print_out += "\n --- batch {} F1 @{}: \t".format(name, topk) \
                             + "{:.4f}".format(results['f_score@{}'.format(topk)])
        else:
            # ignore partial for now
            continue

    print_out += "\n ======================================================="

    return print_out
