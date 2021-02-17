# -*- encoding: utf-8 -*-
import codecs
import json
import random
import shutil

from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser
import os

import datetime
import time
import numpy as np

import kp_evaluate
from onmt.utils import split_corpus
from onmt.utils.logging import init_logger

import onmt.opts as opts


def scan_new_checkpoints(ckpt_dir):
    ckpts = {}
    for subdir, dirs, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith('.pt'):
                ckpt_name = file[: file.find('.pt')]
                ckpts[ckpt_name] = os.path.join(subdir, file)

    return ckpts


def _get_parser():
    parser = ArgumentParser(description='run_kp_eval.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)

    return parser


if __name__ == "__main__":
    parser = _get_parser()

    parser.add_argument('--tasks', '-tasks', nargs='+', type=str,
                        required=True,
                        choices=['pred', 'eval', 'report'],
                        help='Specify process to run, generation or evaluation')
    parser.add_argument('-ckpt_dir', type=str, required=True, help='Directory to all checkpoints')
    parser.add_argument('--step_base', '-step_base', type=int, default=1,
                        help='the base of step to be evaluated, only if ckpt_step % step_base==0 we evaluate it,  '
                             '1 means evaluate everything.')
    parser.add_argument('-output_dir', type=str, required=True, help='Directory to output results')
    parser.add_argument('-data_dir', type=str, required=True, help='Directory to datasets (ground-truth)')
    parser.add_argument('-test_interval', type=int, default=600, help='Minimum time interval the job should wait if a .pred file is not updated by another job (imply another job failed).')
    parser.add_argument('-testsets', nargs='+', type=str, default=["nus", "semeval"], help='Specify datasets to test on')
    # parser.add_argument('-testsets', nargs='+', type=str, default=["kp20k", "duc", "inspec", "krapivin", "nus", "semeval"], help='Specify datasets to test on')
    parser.add_argument('--onepass', '-onepass', action='store_true', help='If true, it only scans and generates once, otherwise an infinite loop scanning new available ckpts.')
    parser.add_argument('--wait_patience', '-wait_patience', type=int, default=1, help='Terminates evaluation after scan this number of times.')
    parser.add_argument('--wait_time', '-wait_time', type=int, default=120, help='.')
    parser.add_argument('--sleep_time', '-sleep_time', type=int, default=600, help='.')
    parser.add_argument('--ignore_existing', '-ignore_existing', action='store_true', help='If true, it ignores previous generated results.')
    parser.add_argument('--eval_topbeam', '-eval_topbeam',action="store_true", help='Evaluate with top beam only (self-terminating) or all beams (full search)')

    opt = parser.parse_args()

    # np.random.seed()
    wait_time = np.random.randint(opt.wait_time)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # "%Y-%m-%d_%H:%M:%S"
    logger = init_logger(opt.output_dir + '/autoeval_%s_%s.log'
                         % ('-'.join(opt.testsets), current_time))
    if not opt.onepass:
        logger.info('Sleep for %d sec to avoid conflicting with other threads' % wait_time)
        time.sleep(wait_time)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(os.path.join(opt.output_dir, 'eval')):
        os.makedirs(os.path.join(opt.output_dir, 'eval'))
    if not os.path.exists(os.path.join(opt.output_dir, 'pred')):
        os.makedirs(os.path.join(opt.output_dir, 'pred'))

    # shutil.copy2(opt.config, opt.output_dir)
    logger.info(opt)

    testset_path_dict = {}
    for testset in opt.testsets:
        src_shard = split_corpus(opt.data_dir + '/%s/%s_test.src' % (testset, testset), shard_size=-1)
        tgt_shard = split_corpus(opt.data_dir + '/%s/%s_test.tgt' % (testset, testset), shard_size=-1)
        src_shard, tgt_shard = list(zip(src_shard, tgt_shard))[0]
        logger.info("Loaded data from %s: #src=%d, #tgt=%d" % (testset, len(src_shard), len(tgt_shard)))
        testset_path_dict[testset] = (opt.data_dir + '/%s/%s_test.src' % (testset, testset),
                                      opt.data_dir + '/%s/%s_test.tgt' % (testset, testset),
                                      src_shard, tgt_shard)

    current_patience = opt.wait_patience
    pred_linecount_dict = {}
    eval_linecount_dict = {}
    while True:
        new_ckpts = scan_new_checkpoints(opt.ckpt_dir)
        new_ckpts_items = sorted(new_ckpts.items(), key=lambda x:int(x[0][x[0].rfind('step_')+5:]))
        random.shuffle(new_ckpts_items)
        logger.info('Found %d checkpoints from %s!' % (len(new_ckpts), opt.ckpt_dir))

        if opt.step_base is not None and opt.step_base > 1:
            logger.warn('-step_base is set, filtering some ckpts')
            new_ckpts_items = [(ckpt_name, ckpt_path) for ckpt_name, ckpt_path in new_ckpts_items if int(ckpt_name[ckpt_name.rfind('step_')+5:]) % opt.step_base == 0 and int(ckpt_name[ckpt_name.rfind('step_')+5:]) // opt.step_base > 0]
            logger.info('After filtering non opt.step_base ckpts, found %d checkpoints!' % (len(new_ckpts_items)))

        job_done = False # a flag indicating if any real pred/eval job is done

        for ckpt_id, (ckpt_name, ckpt_path) in enumerate(new_ckpts_items):
            logger.info("[%d/%d] Checking checkpoint: %s" % (ckpt_id, len(new_ckpts_items), ckpt_path))
            setattr(opt, 'models', [ckpt_path])

            translator = None

            score_dicts = {}
            for dataname, dataset in testset_path_dict.items():
                src_path, tgt_path, src_shard, tgt_shard = dataset

                pred_path = os.path.join(opt.output_dir, 'pred', ckpt_name, '%s.pred' % dataname)
                printout_path = os.path.join(opt.output_dir, 'pred', ckpt_name, '%s.report.txt' % dataname)
                eval_dir = os.path.join(opt.output_dir, 'eval')
                eval_path = os.path.join(eval_dir, ckpt_name + '-%s-%s.json'
                                         % (dataname, 'selfterminating' if opt.eval_topbeam else 'exhaustive'))
                report_path = os.path.join(eval_dir, '%s_summary_%s.csv' % (current_time, '%s'))

                # create dirs
                if not os.path.exists(os.path.join(opt.output_dir, 'pred', ckpt_name)):
                    os.makedirs(os.path.join(opt.output_dir, 'pred', ckpt_name))
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)

                # do translation
                # skip translation for this dataset if previous pred exists
                do_trans_flag = True
                if os.path.exists(pred_path):
                    elapsed_time = time.time() - os.stat(pred_path).st_mtime
                    if pred_path in pred_linecount_dict and pred_linecount_dict[pred_path] == len(src_shard):
                        # if it's already in pred_linecount_dict, means it's done and counted
                        do_trans_flag = False
                    elif elapsed_time < opt.test_interval:
                        do_trans_flag = False
                        logger.info("Skip translating because previous PRED file was generated only %d sec ago (<%d sec)." % (elapsed_time, opt.test_interval))
                    else:
                        # count line numbers of long-done files to check if this is a done pred
                        try:
                            pred_linecount_dict[pred_path] = len([1 for i in open(pred_path, 'r').readlines()])
                            # count is same means it's done
                            if pred_linecount_dict[pred_path] == len(src_shard):
                                do_trans_flag = False
                                logger.info("Skip translating because previous PRED is complete.")
                            else:
                                # if file is modified less than opt.test_interval min, it might be being processed by another job. Otherwise it's a bad result and delete it
                                if elapsed_time < opt.test_interval:
                                    do_trans_flag = False
                                else:
                                    os.remove(pred_path)
                                    logger.info('Removed a bad PRED file, #(line)=%d, #(elapsed_time)=%ds: %s'
                                                % (pred_linecount_dict[pred_path], int(elapsed_time), pred_path))
                        except Exception as e:
                            logger.exception('Error while validating or deleting PRED file: %s' % pred_path)

                if 'pred' in opt.tasks:
                    try:
                        if do_trans_flag or opt.ignore_existing:
                            if translator is None:
                                translator = build_translator(opt, report_score=opt.verbose, logger=logger)
                            # create an empty file to indicate that the translator is working on it
                            codecs.open(pred_path, 'w+', 'utf-8').close()
                            # set output_file for each dataset (instead of outputting to opt.output)
                            translator.out_file = codecs.open(pred_path, 'w+', 'utf-8')
                            logger.info("Start translating [%s] for %s." % (dataname, ckpt_name))
                            logger.info("\t exporting PRED result to %s." % (pred_path))
                            _, _ = translator.translate(
                                src=src_shard,
                                tgt=tgt_shard,
                                src_dir=opt.src_dir,
                                batch_size=opt.batch_size,
                                attn_debug=opt.attn_debug,
                                opt=opt
                            )
                            job_done = True
                        else:
                            logger.info("Skip translating [%s] for %s." % (dataname, ckpt_name))
                    except Exception as e:
                        logger.exception('Error while translating')

                # do evaluation
                do_eval_flag = True
                if not os.path.exists(pred_path):
                    do_eval_flag = False
                    # logger.info("Skip evaluating because no available pred file.")
                else:
                    try:
                        if not pred_path in pred_linecount_dict:
                            pred_linecount_dict[pred_path] = len([1 for i in open(pred_path, 'r').readlines()])
                        num_pred = pred_linecount_dict[pred_path]
                        if num_pred != len(src_shard):
                            do_eval_flag = False
                            logger.info("Skip evaluating because current PRED file is not complete, #(line)=%d." % (num_pred))
                            elapsed_time = time.time() - os.stat(pred_path).st_mtime
                            if elapsed_time > opt.test_interval:
                                os.remove(pred_path)
                                logger.warn('Removed a bad PRED file, #(line)=%d, #(elapsed_time)=%ds: %s' % (num_pred, int(elapsed_time), pred_path))
                        else:
                            # if pred is good, check if re-eval is necessary
                            if os.path.exists(eval_path):
                                elapsed_time = time.time() - os.stat(eval_path).st_mtime
                                if eval_path in eval_linecount_dict and eval_linecount_dict[eval_path] == len(src_shard):
                                    do_eval_flag = False
                                elif elapsed_time < opt.test_interval:
                                    # if file is modified less than opt.test_interval min, it might be being processed by another job.
                                    do_eval_flag = False
                                    logger.info("Skip evaluating because previous EVAL file was generated only %d sec ago (<%d sec)." % (elapsed_time, opt.test_interval))
                                else:
                                    score_dict = json.load(open(eval_path, 'r'))
                                    if 'present_exact_correct@5' in score_dict:
                                        num_eval = len(score_dict['present_exact_correct@5'])
                                    else:
                                        num_eval = 0
                                    eval_linecount_dict[eval_path] = num_eval
                                    if num_eval == len(src_shard):
                                        do_eval_flag = False
                                        logger.info("Skip evaluating because existing eval file is complete.")
                                    else:
                                        # it's a bad result and delete it
                                        os.remove(eval_path)
                                        logger.info('Removed a bad eval file, #(pred)=%d, #(eval)=%d, #(elapsed_time)=%ds: %s' % (num_pred, num_eval, int(elapsed_time), eval_path))
                    except Exception as e:
                        logger.exception('Error while validating or deleting EVAL file: %s' % eval_path)

                if 'eval' in opt.tasks:
                    try:
                        if do_eval_flag or opt.ignore_existing:
                            logger.info("Start evaluating [%s] for %s" % (dataname, ckpt_name))
                            logger.info("\t will export eval result to %s." % (eval_path))
                            score_dict = kp_evaluate.keyphrase_eval(src_path, tgt_path,
                                                                    pred_path=pred_path, logger=logger,
                                                                    verbose=opt.verbose,
                                                                    report_path=printout_path,
                                                                    eval_topbeam=opt.eval_topbeam
                                                                    )
                            if score_dict is not None:
                                score_dicts[dataname] = score_dict
                                with open(eval_path, 'w') as output_json:
                                    output_json.write(json.dumps(score_dict)+'\n')
                            job_done = True
                        else:
                            logger.info("Skip evaluating [%s] for %s." % (dataname, ckpt_name))
                    except Exception as e:
                        logger.exception('Error while evaluating')

                # do generate summarized report
                if 'report' in opt.tasks:
                    kp_evaluate.gather_eval_results(eval_root_dir=eval_dir, report_csv_dir=report_path)

        if job_done: # reset current_patience if no real job is done in the current iteration
            current_patience = opt.wait_patience
        else:
            current_patience -= 1

        if opt.onepass:
            break

        if current_patience <= 0:
            break
        else:
            # scan again for every 10min
            sleep_time = opt.sleep_time
            logger.info('Sleep for %d sec, current_patience=%d' % (sleep_time, current_patience))
            logger.info('*' * 50)
            time.sleep(sleep_time)
