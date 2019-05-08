# -*- encoding: utf-8 -*-
import codecs
import json
import shutil

from onmt.translate.translator import build_translator
from onmt.utils.parse import ArgumentParser
import os

import datetime
import time
import numpy as np
import pandas as pd

import kp_evaluate
from onmt.utils import split_corpus
from onmt.utils.logging import init_logger

import onmt.opts as opts


def scan_new_checkpoints(ckpt_dir, output_dir):
    ckpts = {}
    done_ckpts = {}
    for subdir, dirs, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith('.pt'):
                ckpt_name = file[: file.find('.pt')]
                ckpts[ckpt_name] = os.path.join(subdir, file)

    # for subdir, dirs, files in os.walk(output_dir):
    #     for file in files:
    #         if file.endswith('.score.json'):
    #             ckpt_name = file[: file.find('.score.json')]
    #             done_ckpts[ckpt_name] = os.path.join(subdir, file)
    #
    # for ckpt_name in done_ckpts.keys():
    #     if ckpt_name in ckpts:
    #         del ckpts[ckpt_name]

    return ckpts


def _get_parser():
    parser = ArgumentParser(description='run_kp_eval.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)

    return parser


if __name__ == "__main__":
    parser = _get_parser()

    parser.add_argument('-ckpt_dir', type=str, required=True, help='Directory to all checkpoints')
    parser.add_argument('-output_dir', type=str, required=True, help='Directory to output results')
    parser.add_argument('-data_dir', type=str, required=True, help='Directory to datasets (ground-truth)')
    parser.add_argument('-testsets', nargs='+', type=str, default=["nus", "semeval"], help='Specify datasets to test on')
    parser.add_argument('-test_interval', type=int, default=7200, help='Minimum time interval the job should wait if a .pred file is not updated by another job (imply another job failed).')
    # parser.add_argument('-testsets', nargs='+', type=str, default=["kp20k", "duc", "inspec", "krapivin", "nus", "semeval"], help='Specify datasets to test on')

    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(os.path.join(opt.output_dir, 'eval')):
        os.makedirs(os.path.join(opt.output_dir, 'eval'))
    if not os.path.exists(os.path.join(opt.output_dir, 'pred')):
        os.makedirs(os.path.join(opt.output_dir, 'pred'))
    current_time = datetime.datetime.now().strftime("%Y-%m-%d") # "%Y-%m-%d_%H:%M:%S"
    logger = init_logger(opt.output_dir + 'autoeval_%s_%s.log'
                         % ('-'.join(opt.testsets), current_time))

    shutil.copy2(opt.config, opt.output_dir)
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

    np.random.seed()
    sleep_time = np.random.randint(120)
    logger.info('Sleep for %d sec for avoid conflicting with other threads' % sleep_time)
    time.sleep(sleep_time)
    # if True:
    while True:
        new_ckpts = scan_new_checkpoints(opt.ckpt_dir, opt.output_dir)
        # print(new_ckpts.items())
        # print(sorted(new_ckpts.items(), key=lambda x:int(x[0][x[0].rfind('step_')+5:])))
        for ckpt_name, ckpt_path in sorted(new_ckpts.items(), key=lambda x:int(x[0][x[0].rfind('step_')+5:])):
            logger.info("Processing model from checkpoint: %s" % ckpt_path)
            setattr(opt, 'models', [ckpt_path])

            translator = None

            score_dicts = {}
            for dataname, dataset in testset_path_dict.items():
                src_path, tgt_path, src_shard, tgt_shard = dataset

                if not os.path.exists(os.path.join(opt.output_dir, 'pred', ckpt_name)):
                    os.makedirs(os.path.join(opt.output_dir, 'pred', ckpt_name))

                pred_path = os.path.join(opt.output_dir, 'pred', ckpt_name, '%s.pred' % dataname)
                report_path = os.path.join(opt.output_dir, 'pred', ckpt_name, '%s.report.txt' % dataname)
                score_path = os.path.join(opt.output_dir, 'eval', ckpt_name+'-%s.json' % dataname)

                # skip translation for this dataset if previous pred exists
                do_trans_flag = True
                do_eval_flag = True
                if os.path.exists(pred_path):
                    lines = open(pred_path, 'r').readlines()
                    # count is same means it's done
                    if len(lines) == len(src_shard):
                        do_trans_flag = False
                    # if file is modified less than opt.test_interval min, might be processing by another job
                    elapsed_time = time.time() - os.stat(pred_path).st_mtime
                    if elapsed_time < opt.test_interval:
                        do_trans_flag = False
                        do_eval_flag = False

                if do_trans_flag:
                    if translator is None:
                        translator = build_translator(opt, report_score=opt.verbose, logger=logger)
                    # create an empty file to indicate that the translator is working on it
                    codecs.open(pred_path, 'w+', 'utf-8').close()
                    # set output_file for each dataset (instead of outputting to opt.output)
                    translator.out_file = codecs.open(pred_path, 'w+', 'utf-8')
                    logger.info("Start translating %s." % dataname)
                    _, _ = translator.translate(
                        src=src_shard,
                        tgt=tgt_shard,
                        src_dir=opt.src_dir,
                        batch_size=opt.batch_size,
                        attn_debug=opt.attn_debug,
                        opt=opt
                    )

                if do_eval_flag:
                    logger.info("Start evaluating generated results of %s" % ckpt_name)
                    score_dict = kp_evaluate.keyphrase_eval(src_path, tgt_path,
                                                            pred_path=pred_path, logger=logger,
                                                            verbose=opt.verbose,
                                                            report_path=report_path)
                    score_dicts[dataname] = score_dict

                    with open(score_path, 'w') as output_json:
                        output_json.write(json.dumps(score_dict))

        kp_evaluate.export_summary_to_csv(json_root_dir=os.path.join(opt.output_dir, 'eval'), output_csv_path=os.path.join(opt.output_dir, '%s_summary_%s.csv' % (current_time, '%s')))

        # scan again for every 5min
        sleep_time = 600
        logger.info('Sleep for %d sec' % sleep_time)
        time.sleep(sleep_time)
