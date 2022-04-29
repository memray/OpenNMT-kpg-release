# -*- encoding: utf-8 -*-
import codecs
import json
import random
import shutil

from onmt.constants import ModelTask
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

train_test_mappings = {
    'kp20k': ['kp20k', 'kp20k_valid2k', 'inspec', 'krapivin', 'semeval', 'nus', 'duc'],
    'openkp': ['openkp', 'openkp_valid2k', 'jptimes', 'duc'],
    'kptimes': ['kptimes', 'kptimes_valid2k', 'jptimes', 'duc'],
    'stackex': ['stackex', 'stackex_valid2k', 'duc'],
}

def scan_new_checkpoints(ckpt_dir):
    ckpts = []
    for subdir, dirs, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith('.pt'):
                ckpt_name = file[: file.find('.pt')]
                if not 'step' in ckpt_name:
                    # skip the last ckpt of fairseq jobs such as 'checkpoint_last.pt' and 'checkpoint54.pt'
                    continue
                ckpt_path = os.path.join(subdir, file)
                exp_dir = os.path.dirname(subdir)
                exp_name = exp_dir[exp_dir.rfind('/') + 1: ]
                ckpt_dict = {
                    'exp_dir': exp_dir,
                    'exp_name': exp_name,
                    'ckpt_path': ckpt_path,
                    'ckpt_name': ckpt_name,
                    'step': int(ckpt_name[ckpt_name.rfind('step_')+5:])
                }
                ckpts.append(ckpt_dict)

    return ckpts

def scan_predictions(exp_root_dir):
    preds = []
    for subdir, dirs, files in os.walk(exp_root_dir):
        for file in files:
            if file.endswith('.pred'):
                try:
                    pred_name = file[: -5]
                    step, data = pred_name.split('-')
                    step = int(step[step.rfind('step_') + 5:])
                    data = data[5:]
                    pred = {
                        'ckpt_name': pred_name,
                        'pred_path': os.path.join(subdir, file),
                        'dataset': data,
                        'step': step
                    }
                    preds.append(pred)
                except:
                    print('invalid pred name %s' % file)

    return preds

def _get_parser():
    parser = ArgumentParser(description='run_kp_eval_transfer.py')

    opts.translate_opts(parser)
    # opts.train_opts(parser)
    opts.model_opts(parser)
    opts.dynamic_prepare_opts(parser, build_vocab_only=False)

    return parser


if __name__ == "__main__":
    parser = _get_parser()

    parser.add_argument('--tasks', '-tasks', nargs='+', type=str,
                        required=True,
                        choices=['pred', 'eval', 'report'],
                        help='Specify process to run, generation or evaluation')
    parser.add_argument('-exp_root_dir', type=str, required=True, help='Directory to all checkpoints')
    parser.add_argument('-data_dir', type=str, required=True, help='Directory to datasets (ground-truth)')
    parser.add_argument('--step_base', '-step_base', type=int, default=1,
                        help='the base of step to be evaluated, only if ckpt_step % step_base==0 we evaluate it,  '
                             '1 means evaluate everything.')
    parser.add_argument('-test_interval', type=int, default=600, help='Minimum time interval the job should wait if a .pred file is not updated by another job (imply another job failed).')
    parser.add_argument('-testsets', nargs='+', type=str, default=['kp20k', 'openkp', 'stackex', 'kptimes', 'jptimes'], help='Specify datasets to test on')
    parser.add_argument('--pred_trained_only', '-pred_trained_only', action='store_true', help='If true, it only runs inference of testsets that the job has been trained with.')
    parser.add_argument('--ignore_existing', '-ignore_existing', action='store_true', help='If true, it ignores previous generated results.')
    parser.add_argument('--empirical_result', '-empirical_result', action='store_true', help='For export empirical results to CSV.')

    parser.add_argument('-splits', nargs='+', type=str, default=['train', 'test', 'valid'], help='Specify datasets to test on')
    parser.add_argument('-tokenizer', type=str, default='split_nopunc', choices=['spacy', 'split', 'split_nopunc'],
                        help='Specify what tokenizer used in evaluation')

    parser.add_argument('--onepass', '-onepass', action='store_true', help='If true, it only scans and generates once, otherwise an infinite loop scanning new available ckpts.')
    parser.add_argument('--wait_patience', '-wait_patience', type=int, default=2, help='Terminates evaluation after scan this number of times.')
    parser.add_argument('--wait_time', '-wait_time', type=int, default=120, help='.')
    parser.add_argument('--sleep_time', '-sleep_time', type=int, default=600, help='.')

    opt = parser.parse_args()
    if isinstance(opt.data, str):
        setattr(opt, 'data', json.loads(opt.data.replace('\'', '"')))
    setattr(opt, 'data_task', ModelTask.SEQ2SEQ)
    if opt.data:
        ArgumentParser._get_all_transform(opt)
        ArgumentParser._validate_transforms_opts(opt)
        ArgumentParser._validate_fields_opts(opt)

    opt.__setattr__('valid_batch_size', opt.batch_size)
    opt.__setattr__('batch_size_multiple', 1)
    opt.__setattr__('bucket_size', 128)
    opt.__setattr__('pool_factor', 256)

    # np.random.seed()
    wait_time = np.random.randint(opt.wait_time) if opt.wait_time > 0 else 0
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # "%Y-%m-%d_%H:%M:%S"
    if not os.path.exists(os.path.join(opt.exp_root_dir, 'logs')):
        os.makedirs(os.path.join(opt.exp_root_dir, 'logs'))
    logger = init_logger(opt.exp_root_dir + '/logs/autoeval_%s.log'
                         % (current_time))
    if not opt.onepass:
        logger.info('Sleep for %d sec to avoid conflicting with other threads' % wait_time)
        time.sleep(wait_time)

    # do generate summarized report
    if 'report' in opt.tasks:
        report_dir = os.path.join(opt.exp_root_dir, 'report')
        if not os.path.exists(report_dir): os.makedirs(report_dir)
        kp_evaluate.gather_eval_results(eval_root_dir=opt.exp_root_dir,
                                        report_csv_dir=report_dir,
                                        tokenizer=opt.tokenizer,
                                        empirical_result=opt.empirical_result)
        logger.warning('Report accomplished, exit!')
        exit(0)

    logger.info(opt)
    testset_path_dict = {}
    for testset in opt.testsets:
        for split in opt.splits:
            if not os.path.exists(opt.data_dir + '/%s/%s.json' % (testset, split)):
                logger.info("Data does not exist, skip: %s-%s" % (testset, split))
                continue
            src_shard = split_corpus(opt.data_dir + '/%s/%s.json' % (testset, split), shard_size=-1)
            tgt_shard = split_corpus(opt.data_dir + '/%s/%s.json' % (testset, split), shard_size=-1)
            src_shard, tgt_shard = list(zip(src_shard, tgt_shard))[0]
            src_shard = [json.loads(l) for l in src_shard]
            tgt_shard = [json.loads(l) for l in tgt_shard]
            logger.info("Loaded data from %s-%s: #src=%d, #tgt=%d" % (testset, split, len(src_shard), len(tgt_shard)))
            testset_path_dict[testset+'_'+split] = (opt.data_dir + '/%s/%s.json' % (testset, split),
                                          opt.data_dir + '/%s/%s.json' % (testset, split),
                                          src_shard, tgt_shard)

    current_patience = opt.wait_patience

    while True:
        pred_linecount_dict = {}
        eval_linecount_dict = {}
        job_done = False # a flag indicating if any real pred/eval job is done

        if 'pred' in opt.tasks:
            ckpts = scan_new_checkpoints(opt.exp_root_dir)
            ckpts = sorted(ckpts, key=lambda x:x['step'])
            random.shuffle(ckpts)
            logger.info('Found %d checkpoints from %s!' % (len(ckpts), opt.exp_root_dir))

            if opt.step_base is not None and opt.step_base > 1:
                num_ckpts = len(ckpts)
                ckpts = [ckpt for ckpt in ckpts if ckpt['step'] % opt.step_base == 0 and ckpt['step'] // opt.step_base > 0]
                logger.info('After filtering non opt.step_base=%d ckpts, found %d/%d checkpoints!' %
                            (opt.step_base, len(ckpts), num_ckpts))

            # iterate each ckpt and start predicting/evaluating
            for ckpt_id, ckpt in enumerate(ckpts):
                ckpt_path = ckpt['ckpt_path']
                ckpt_name = ckpt['ckpt_name']
                exp_dir = ckpt['exp_dir']
                exp_name = ckpt['exp_name']
                logger.info("[%d/%d] Checking checkpoint: %s" % (ckpt_id, len(ckpts), ckpt_path))
                setattr(opt, 'models', [ckpt['ckpt_path']])

                for datasplit_name, dataset in testset_path_dict.items():
                    # ignore current testdata-split if this data is not used in training
                    if opt.pred_trained_only:
                        pass_flag = False
                        cur_testname = datasplit_name[: datasplit_name.index('_')] if '_' in datasplit_name else datasplit_name
                        for trainname, testnames in train_test_mappings.items():
                            # training dataset name appears in exp_name
                            if trainname in exp_name and cur_testname in testnames:
                                pass_flag = True
                                break
                        if not pass_flag:
                            print('Skip predict/evaluate test=[%s] for ckpt=[%s] due to train/test mismatch.' % (datasplit_name, exp_name))
                            continue

                    src_path, tgt_path, src_shard, tgt_shard = dataset

                    decoding_method = 'beamsearch-width_%d-maxlen_%d' % (opt.beam_size, opt.max_length)
                    pred_dir = os.path.join(exp_dir, 'outputs', decoding_method, 'pred')
                    if not os.path.exists(pred_dir): os.makedirs(pred_dir)
                    pred_file = '%s-data_%s.pred' % (ckpt_name, datasplit_name)
                    pred_path = os.path.join(pred_dir, pred_file)

                    # skip translation for this dataset if previous pred exists
                    do_trans_flag = True
                    if os.path.exists(pred_path):
                        elapsed_time = time.time() - os.stat(pred_path).st_mtime
                        if pred_path in pred_linecount_dict and pred_linecount_dict[pred_path] == len(src_shard):
                            # if it's already in pred_linecount_dict, means it's done and counted
                            do_trans_flag = False
                        elif elapsed_time < opt.test_interval:
                            do_trans_flag = False
                            logger.info("Skip translating because previous PRED file was generated only %d sec ago (<%d sec). PRED file: %s"
                                        % (elapsed_time, opt.test_interval, pred_path))
                        else:
                            # count line numbers of long-done files to check if this is a done pred
                            try:
                                pred_linecount_dict[pred_path] = len([1 for i in open(pred_path, 'r').readlines()])
                                # count is same means it's done
                                if pred_linecount_dict[pred_path] == len(src_shard):
                                    do_trans_flag = False
                                    logger.info("Skip translating because previous PRED is complete. PRED file: %s" % (pred_path))
                                else:
                                    # if file is modified less than opt.test_interval min, it might be being processed by another job. Otherwise it's a bad result and delete it
                                    if elapsed_time < opt.test_interval:
                                        do_trans_flag = False
                                    else:
                                        os.remove(pred_path)
                                        logger.info('Removed a bad PRED file, #(line)=%d, #(elapsed_time)=%ds. PRED file: %s'
                                                    % (pred_linecount_dict[pred_path], int(elapsed_time), pred_path))
                            except Exception as e:
                                logger.exception('Error while validating or deleting PRED file: %s' % pred_path)

                    opt.data['valid']['path_src'] = src_path
                    opt.data['valid']['path_tgt'] = src_path
                    opt.data['valid']['path_align'] = None

                    # do translation
                    try:
                        if do_trans_flag or opt.ignore_existing:
                            logger.info("*" * 50)
                            logger.info("Start translating [data=%s] for [exp=%s]-[ckpt=%s]." % (datasplit_name, exp_name, ckpt_name))
                            logger.info("\t exporting PRED result to %s." % (pred_path))
                            logger.info("*" * 50)

                            # if it's BART model, OpenNMT has to do something additional
                            if 'bart' in exp_name.lower() and not exp_name.lower().startswith('transformer'):
                                opt.__setattr__('fairseq_model', True)
                                opt.__setattr__('encoder_type', 'bart')
                                opt.__setattr__('decoder_type', 'bart')
                                opt.__setattr__('pretrained_tokenizer', True)
                                opt.__setattr__('copy_attn', False)
                                opt.__setattr__('model_dtype', 'fp16')
                            else:
                                opt.__setattr__('fairseq_model', False)

                            translator = build_translator(opt, report_score=opt.verbose, logger=logger)
                            # create an empty file to indicate that the translator is working on it
                            codecs.open(pred_path, 'w+', 'utf-8').close()
                            # set output_file for each dataset (instead of outputting to opt.output)
                            translator.out_file = codecs.open(pred_path, 'w+', 'utf-8')
                            _, _ = translator.translate(
                                src=src_shard,
                                batch_size=opt.batch_size,
                                attn_debug=opt.attn_debug,
                                opt=opt
                            )
                            job_done = True
                            logger.info("Complete translating [%s], PRED file: %s." % (datasplit_name, pred_path))
                        else:
                            logger.info("Skip translating [%s] for %s, PRED file: %s." % (datasplit_name, ckpt_name, pred_path))
                    except Exception as e:
                        logger.exception('Error while translating [%s], PRED file: %s.' % (datasplit_name, pred_path))

        # do evaluation
        if 'eval' in opt.tasks:
            new_preds = scan_predictions(opt.exp_root_dir)
            new_preds = sorted(new_preds, key=lambda x: x['step'])
            random.shuffle(new_preds)
            logger.info('Found %d predictions from %s!' % (len(new_preds), opt.exp_root_dir))

            for pred_id, pred in enumerate(new_preds):
                ckpt_name = pred['ckpt_name']
                pred_path = pred['pred_path']
                datasplit_name = pred['dataset']

                if datasplit_name not in testset_path_dict: continue

                logger.info("[%d/%d] Checking prediction: %s" % (pred_id, len(new_preds), pred_path))
                eval_path = pred_path[: -5] + '.%s.eval' % opt.tokenizer
                printout_path = pred_path[: -5] + '.%s.report' % opt.tokenizer

                do_eval_flag = True
                try:
                    src_path, tgt_path, src_shard, tgt_shard = testset_path_dict[datasplit_name]
                    if not pred_path in pred_linecount_dict: # may be out of date
                        pred_linecount_dict[pred_path] = len([1 for i in open(pred_path, 'r').readlines()])
                    num_pred = pred_linecount_dict[pred_path]
                    if num_pred != len(src_shard):
                        do_eval_flag = False
                        logger.info("Skip evaluating because current PRED file is not complete, #(line)=%d. PRED file: %s"
                                    % (num_pred, pred_path))
                        elapsed_time = time.time() - os.stat(pred_path).st_mtime
                        if elapsed_time > opt.test_interval:
                            os.remove(pred_path)
                            logger.warning('Removed a bad PRED file, #(line)=%d, #(elapsed_time)=%ds. PRED file: %s'
                                        % (num_pred, int(elapsed_time), pred_path))
                            del pred_linecount_dict[pred_path]
                    else:
                        # if pred is good, check if re-eval is necessary
                        if os.path.exists(eval_path):
                            elapsed_time = time.time() - os.stat(eval_path).st_mtime
                            if eval_path in eval_linecount_dict and eval_linecount_dict[eval_path] == len(src_shard):
                                do_eval_flag = False
                            elif elapsed_time < opt.test_interval:
                                # if file is modified less than opt.test_interval min, it might be being processed by another job.
                                do_eval_flag = False
                                logger.info("Skip evaluating because previous EVAL file was generated only %d sec ago (<%d sec). EVAL file: %s"
                                            % (elapsed_time, opt.test_interval, eval_path))
                            else:
                                try:
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
                                except:
                                    os.remove(eval_path)
                                    logger.info('Removed a bad eval file: %s' % (eval_path))
                except Exception as e:
                    logger.exception('Error while validating or deleting EVAL file: %s' % eval_path)

                try:
                    if do_eval_flag or opt.ignore_existing:
                        logger.info("*" * 50)
                        logger.info("Start evaluating [%s] for %s" % (datasplit_name, ckpt_name))
                        logger.info("\t will export eval result to %s." % (eval_path))
                        logger.info("*" * 50)
                        codecs.open(eval_path, 'w+', 'utf-8').close()

                        score_dict = kp_evaluate.keyphrase_eval(datasplit_name,
                                                                src_path, tgt_path,
                                                                pred_path=pred_path, logger=logger,
                                                                verbose=opt.verbose,
                                                                report_path=printout_path,
                                                                tokenizer=opt.tokenizer
                                                                )
                        if score_dict is not None:
                            with open(eval_path, 'w') as output_json:
                                output_json.write(json.dumps(score_dict)+'\n')
                        job_done = True
                    else:
                        logger.info("Skip evaluating [%s] for %s, EVAL file: %s." % (datasplit_name, ckpt_name, eval_path))
                except Exception as e:
                    logger.exception('Error while evaluating')

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
