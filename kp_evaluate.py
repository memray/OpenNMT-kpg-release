
import argparse
import json
import time

import tqdm
import numpy as np
from collections import Counter

import pandas as pd

import os

from onmt.inputters.keyphrase_dataset import infer_dataset_type, KP_DATASET_FIELDS, parse_src_fn
from onmt.keyphrase import utils
from onmt.keyphrase.eval import compute_match_scores, run_classic_metrics, run_advanced_metrics
from onmt.keyphrase.utils import if_present_duplicate_phrases, validate_phrases, print_predeval_result, gather_scores
from onmt.utils.logging import init_logger


def evaluate(src_list, tgt_list, pred_list,
             unk_token,
             logger=None, verbose=False,
             report_path=None, eval_topbeam=False):
    # progbar = Progbar(logger=logger, title='', target=len(pred_list), total_examples=len(pred_list))

    if report_path:
        report_file = open(report_path, 'w+')
    else:
        report_file = None
    # 'k' means the number of phrases in ground-truth
    topk_range = [5, 10, 'k', 'M']
    absent_topk_range = [10, 50, 'M']
    # 'precision_hard' and 'f_score_hard' mean that precision is calculated with denominator strictly as K (say 5 or 10), won't be lessened even number of preds is smaller
    metric_names = ['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard']

    individual_score_dicts = []  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}
    gathered_score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    # for i, (src_dict, tgt_dict, pred_dict) in tqdm.tqdm(enumerate(zip(src_list, tgt_list, pred_list))):
    for i, (src_dict, tgt_dict, pred_dict) in tqdm.tqdm(enumerate(zip(src_list, tgt_list, pred_list))):
        """
        1. Process each data example and predictions
        """
        src_seq = src_dict["src"].split()
        tgt_seqs = [t.split() for t in tgt_dict["tgt"]]

        if eval_topbeam:
            pred_seqs = pred_dict["topseq_pred_sents"]
            pred_idxs = pred_dict["topseq_preds"] if "topseq_preds" in pred_dict else None
            pred_scores = pred_dict["topseq_pred_scores"] if "topseq_pred_scores" in pred_dict else None
            copied_flags = None
        else:
            pred_seqs = pred_dict["pred_sents"]
            pred_idxs = pred_dict["preds"] if "preds" in pred_dict else None
            pred_scores = pred_dict["pred_scores"] if "pred_scores" in pred_dict else None
            copied_flags = pred_dict["copied_flags"] if "copied_flags" in pred_dict else None

        # 1st filtering, ignore phrases having <unk> and puncs
        valid_pred_flags = validate_phrases(pred_seqs, unk_token)
        # 2nd filtering: filter out phrases that don't appear in text, and keep unique ones after stemming
        present_pred_flags, _, duplicate_flags = if_present_duplicate_phrases(src_seq, pred_seqs)
        # treat duplicates as invalid
        valid_pred_flags = valid_pred_flags * ~duplicate_flags if len(valid_pred_flags) > 0 else []
        valid_and_present_flags = valid_pred_flags * present_pred_flags if len(valid_pred_flags) > 0 else []
        valid_and_absent_flags = valid_pred_flags * ~present_pred_flags if len(valid_pred_flags) > 0 else []

        # compute match scores (exact, partial and mixed), for exact it's a list otherwise matrix
        match_scores_exact = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs, type='exact')
        match_scores_partial = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs, type='ngram')
        # simply add full-text to n-grams might not be good as its contribution is not clear
        # match_scores_mixed = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs, type='mixed')

        # split tgts by present/absent
        present_tgt_flags, _, _ = if_present_duplicate_phrases(src_seq, tgt_seqs)
        present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
        absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if ~present]

        # filter out results of invalid preds
        valid_preds = [seq for seq, valid in zip(pred_seqs, valid_pred_flags) if valid]
        valid_present_pred_flags = present_pred_flags[valid_pred_flags]

        match_scores_exact = match_scores_exact[valid_pred_flags]
        match_scores_partial = match_scores_partial[valid_pred_flags]
        # match_scores_mixed = match_scores_mixed[valid_pred_flags]

        # split preds by present/absent and exact/partial/mixed
        valid_present_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if present]
        valid_absent_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if ~present]
        if len(valid_present_pred_flags) > 0:
            present_exact_match_scores = match_scores_exact[valid_present_pred_flags]
            present_partial_match_scores = match_scores_partial[valid_present_pred_flags][:, present_tgt_flags]
            # present_mixed_match_scores = match_scores_mixed[present_pred_flags][:, present_tgt_flags]
            absent_exact_match_scores = match_scores_exact[~valid_present_pred_flags]
            absent_partial_match_scores = match_scores_partial[~valid_present_pred_flags][:, ~present_tgt_flags]
            # absent_mixed_match_scores = match_scores_mixed[~present_pred_flags][:, ~present_tgt_flags]
        else:
            present_exact_match_scores = []
            present_partial_match_scores = []
            # present_mixed_match_scores = []
            absent_exact_match_scores = []
            absent_partial_match_scores = []
            # absent_mixed_match_scores = []

        # assert len(valid_pred_seqs) == len(match_scores_exact) == len(present_pred_flags)
        # assert len(present_preds) == len(present_exact_match_scores) == len(present_partial_match_scores) == len(present_mixed_match_scores)
        # assert present_partial_match_scores.shape == present_mixed_match_scores.shape
        # assert len(absent_preds) == len(absent_exact_match_scores) == len(absent_partial_match_scores) == len(absent_mixed_match_scores)
        # assert absent_partial_match_scores.shape == absent_mixed_match_scores.shape


        """
        2. Compute metrics
        """
        # get the scores on different scores (for absent results, only recall matters)
        present_exact_results = run_classic_metrics(present_exact_match_scores, valid_present_preds, present_tgts, metric_names, topk_range)
        absent_exact_results = run_classic_metrics(absent_exact_match_scores, valid_absent_preds, absent_tgts, metric_names, absent_topk_range)
        present_partial_results = run_classic_metrics(present_partial_match_scores, valid_present_preds, present_tgts, metric_names, topk_range, type='partial')
        absent_partial_results = run_classic_metrics(absent_partial_match_scores, valid_absent_preds, absent_tgts, metric_names, absent_topk_range, type='partial')
        # present_mixed_results = run_metrics(present_mixed_match_scores, present_preds, present_tgts, metric_names, topk_range, type='partial')
        # absent_mixed_results = run_metrics(absent_mixed_match_scores, absent_preds, absent_tgts, metric_names, absent_topk_range, type='partial')

        present_exact_advanced_results = run_advanced_metrics(present_exact_match_scores, valid_present_preds, present_tgts)
        absent_exact_advanced_results = run_advanced_metrics(absent_exact_match_scores, valid_absent_preds, absent_tgts)
        # print(advanced_present_exact_results)
        # print(advanced_absent_exact_results)

        """
        3. Gather scores
        """
        eval_results_names = [
            'present_exact', 'absent_exact',
            'present_partial', 'absent_partial',
            # 'present_mixed', 'absent_mixed'
            'present_exact_advanced', 'absent_exact_advanced',
            ]
        eval_results_list = [present_exact_results, absent_exact_results,
                             present_partial_results, absent_partial_results,
                             # present_mixed_results, absent_mixed_results
                             present_exact_advanced_results, absent_exact_advanced_results
                            ]
        # update score_dict, appending new scores (results_list) to it
        individual_score_dict = {result_name: results for result_name, results in zip(eval_results_names, eval_results_list)}
        gathered_score_dict = gather_scores(gathered_score_dict, eval_results_names, eval_results_list)

        # add tgt/pred count for computing average performance on non-empty items
        stats_results_names = ['present_tgt_num', 'absent_tgt_num', 'present_pred_num', 'absent_pred_num', 'unique_pred_num', 'dup_pred_num', 'beam_num', 'beamstep_num']
        stats_results_list = [
                        {'present_tgt_num': len(present_tgts)},
                        {'absent_tgt_num': len(absent_tgts)},
                        {'present_pred_num': len(valid_present_preds)},
                        {'absent_pred_num': len(valid_absent_preds)},
                        # TODO some stat should be calculated here since exhaustive/self-terminating makes difference
                        {'unique_pred_num': pred_dict['unique_pred_num'] if 'unique_pred_num' in pred_dict else 0},
                        {'dup_pred_num': pred_dict['dup_pred_num'] if 'dup_pred_num' in pred_dict else 0},
                        {'beam_num': pred_dict['beam_num'] if 'beam_num' in pred_dict else 0},
                        {'beamstep_num': pred_dict['beamstep_num'] if 'beamstep_num' in pred_dict else 0},
                        ]
        for result_name, result_dict in zip(stats_results_names, stats_results_list):
            individual_score_dict[result_name] = result_dict[result_name]
        gathered_score_dict = gather_scores(gathered_score_dict, stats_results_names, stats_results_list)
        # individual_score_dicts.append(individual_score_dict)

        """
        4. Print results if necessary
        """
        if verbose or report_file:
            print_out = print_predeval_result(i, src_dict,
                                              tgt_seqs, present_tgt_flags,
                                              pred_seqs, pred_scores, pred_idxs, copied_flags,
                                              present_pred_flags, valid_pred_flags,
                                              valid_and_present_flags, valid_and_absent_flags,
                                              match_scores_exact, match_scores_partial,
                                              eval_results_names, eval_results_list, gathered_score_dict)

            if verbose:
                if logger:
                    logger.info(print_out)
                else:
                    print(print_out)

            if report_file:
                report_file.write(print_out)

    # for k, v in score_dict.items():
    #     print('%s, num=%d, mean=%f' % (k, len(v), np.average(v)))

    if report_file:
        report_file.close()

    return gathered_score_dict


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


def keyphrase_eval(src_path, tgt_path, pred_path,
                   unk_token='<unk>', verbose=False, logger=None,
                   report_path=None, eval_topbeam=False, model_name='nn'):
    # change data loader to iterator, otherwise it consumes more than 64gb RAM
    # check line numbers first
    src_line_number = sum([1 for _ in open(src_path, "r")])
    tgt_line_number = sum([1 for _ in open(tgt_path, "r")])
    if model_name == 'nn':
        pred_line_number = sum([1 for _ in open(pred_path, "r")])
    else:
        pred_line_number = len(baseline_pred_loader(pred_path, model_name))

    logger.info("pred file=%s" % (pred_path))
    logger.info("#(src)=%d, #(tgt)=%d, #(pred)=%d" % (src_line_number, tgt_line_number, pred_line_number))
    if src_line_number == tgt_line_number == pred_line_number:
        src_data = [json.loads(l) for l in open(src_path, "r")]
        tgt_data = [json.loads(l) for l in open(tgt_path, "r")]

        # Load from the json-format raw data, preprocess the src and tgt
        if src_path.endswith('json') or src_path.endswith('jsonl'):
            assert src_path == tgt_path, \
                'src and tgt should be from the same raw file: \n\tsrc_path: %s \n\ttgt_path: %s' % (src_path, tgt_path)
            dataset_type = infer_dataset_type(src_path)
            title_field, text_field, keyword_field, _ = KP_DATASET_FIELDS[dataset_type]

            for src_ex, tgt_ex in zip(src_data, tgt_data):
                src_str = parse_src_fn(src_ex, title_field, text_field)
                src_tokens = utils.meng17_tokenize(src_str)
                if isinstance(tgt_ex[keyword_field], str):
                    tgt_kps = tgt_ex[keyword_field].split(';')
                else:
                    tgt_kps = tgt_ex[keyword_field]
                tgt_kps_tokens = [utils.meng17_tokenize(tgt_str) for tgt_str in tgt_kps]
                src_ex['src'] = ' '.join(src_tokens)
                tgt_ex['tgt'] = [' '.join(tgt_tokens) for tgt_tokens in tgt_kps_tokens]
        if model_name == 'nn':
            pred_data = [json.loads(l) for l in open(pred_path, "r")]
        else:
            pred_data = baseline_pred_loader(pred_path, model_name)
        start_time = time.time()
        results_dict = evaluate(src_data, tgt_data, pred_data,
                                unk_token=unk_token,
                                logger=logger, verbose=verbose,
                                report_path=report_path, eval_topbeam=eval_topbeam)
        total_time = time.time() - start_time
        logger.info("Total evaluation time (s): %f" % total_time)

        return results_dict
    else:
        logger.error("")
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


def gather_eval_results(eval_root_dir, report_csv_path=None):
    dataset_scores_dict = {}

    # total_file_num = len(*[[file for file in files if file.endswith('.json')] for subdir, dirs, files in os.walk(json_root_dir)])
    # file_count = 0

    for subdir, dirs, files in os.walk(eval_root_dir):
        for file in files:
            if not file.endswith('.eval'):
                continue
            # file_count += 1
            # print("file_count/file_num=%d/%d" % (file_count, total_file_num))

            file_name = file[: file.find('.eval')]
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

    if report_csv_path:
        for dataset, score_df in dataset_scores_dict.items():
            print("Writing summary to: %s" % (report_csv_path % dataset))
            score_df.to_csv(report_csv_path % dataset)

    return dataset_scores_dict

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

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = init_opt()
    score_dicts = {}

    for ckpt_name in os.listdir(opt.pred_dir):
        if not os.path.isdir(os.path.join(opt.pred_dir, ckpt_name)):
            continue

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

        gather_eval_results(eval_root_dir=os.path.join(opt.output_dir, 'eval'),
                            report_csv_path=os.path.join(opt.output_dir, 'summary_%s.csv' % ('%s')))

        logger.info("Done!")