# -*- encoding: utf-8 -*-
import codecs
import json
import random
import re
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
    parser.add_argument('-output_dir', type=str, required=True, help='Directory to outputs')
    parser.add_argument('-data_dir', type=str, required=True, help='Directory to datasets (ground-truth)')
    parser.add_argument('--wait_patience', '-wait_patience', type=int, default=2, help='Terminates evaluation after scan this number of times.')
    parser.add_argument('--wait_time', '-wait_time', type=int, default=120, help='.')
    parser.add_argument('--sleep_time', '-sleep_time', type=int, default=600, help='.')
    parser.add_argument('-test_interval', type=int, default=600,
                        help='Minimum time interval the job should wait if a .pred file is not updated by another job (imply another job failed).')

    opt = parser.parse_args()
    if isinstance(opt.data, str):
        setattr(opt, 'data', json.loads(opt.data.replace('\'', '"')))
    setattr(opt, 'data_task', ModelTask.SEQ2SEQ)
    if opt.data: ArgumentParser._get_all_transform(opt)

    opt.__setattr__('valid_batch_size', opt.batch_size)
    opt.__setattr__('batch_size_multiple', 1)
    opt.__setattr__('bucket_size', 128)
    opt.__setattr__('pool_factor', 256)

    # np.random.seed()
    wait_time = np.random.randint(opt.wait_time) if opt.wait_time > 0 else 0
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # "%Y-%m-%d_%H:%M:%S"
    if not os.path.exists(os.path.join(opt.exp_root_dir, 'logs')):
        os.makedirs(os.path.join(opt.exp_root_dir, 'logs'))
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    logger = init_logger(opt.exp_root_dir + '/logs/autoeval_%s.log'
                         % (current_time))

    logger.info(opt)

    current_patience = opt.wait_patience
    pred_linecount_dict = {}
    eval_linecount_dict = {}

    while True:
        job_done = False # a flag indicating if any real pred/eval job is done

        ckpts = scan_new_checkpoints(opt.exp_root_dir)
        ckpts = sorted(ckpts, key=lambda x:x['step'])
        random.shuffle(ckpts)
        logger.info('Found %d checkpoints from %s!' % (len(ckpts), opt.exp_root_dir))

        # iterate each ckpt and start predicting/evaluating
        for ckpt_id, ckpt in enumerate(ckpts):
            ckpt_path = ckpt['ckpt_path']
            ckpt_name = ckpt['ckpt_name']
            exp_dir = ckpt['exp_dir']
            exp_name = ckpt['exp_name']
            logger.info("[%d/%d] Checking checkpoint: %s" % (ckpt_id, len(ckpts), ckpt_path))
            setattr(opt, 'models', [ckpt['ckpt_path']])

            file_list = os.listdir(opt.data_dir)
            random.shuffle(file_list)
            for src_file in file_list:
                # ignore current testdata-split if this data is not used in training
                if not re.match('train_\d+.json', src_file):
                    continue
                src_path = os.path.join(opt.data_dir, src_file)
                pred_path = os.path.join(opt.output_dir, src_file)
                data_line_number = 58583 if src_file == 'train_120.json' else 100000

                # skip translation for this dataset if previous pred exists
                do_trans_flag = True
                if os.path.exists(pred_path):
                    elapsed_time = time.time() - os.stat(pred_path).st_mtime
                    if pred_path in pred_linecount_dict and pred_linecount_dict[pred_path] == data_line_number:
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
                            # counts are same means it's done
                            if pred_linecount_dict[pred_path] == data_line_number:
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
                    if do_trans_flag:
                        logger.info("*" * 50)
                        logger.info("Start translating [data=%s] for [exp=%s]-[ckpt=%s]." % (src_file, exp_name, ckpt_name))
                        logger.info("\t exporting PRED result to %s." % (pred_path))
                        logger.info("*" * 50)

                        # if it's BART model, OpenNMT has to do something additional
                        if 'bart' in exp_name:
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
                        src_shard = [json.loads(l) for l in open(src_path, 'r')]
                        _, _ = translator.translate(
                            src=src_shard,
                            batch_size=opt.batch_size,
                            attn_debug=opt.attn_debug,
                            opt=opt
                        )
                        job_done = True
                        logger.info("Complete translating [%s], PRED file: %s." % (src_file, pred_path))
                    else:
                        logger.info("Skip translating [%s] for %s, PRED file: %s." % (src_file, ckpt_name, pred_path))
                except Exception as e:
                    logger.exception('Error while translating [%s], PRED file: %s.' % (src_file, pred_path))

        if job_done: # reset current_patience if no real job is done in the current iteration
            current_patience = opt.wait_patience
        else:
            current_patience -= 1

        if current_patience <= 0:
            break
        else:
            # scan again for every 10min
            sleep_time = opt.sleep_time
            logger.info('Sleep for %d sec, current_patience=%d' % (sleep_time, current_patience))
            logger.info('*' * 50)
            time.sleep(sleep_time)
