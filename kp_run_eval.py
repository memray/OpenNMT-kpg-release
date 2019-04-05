# -*- encoding: utf-8 -*-
import codecs
import json

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

    for subdir, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.score.json'):
                ckpt_name = file[: file.find('.score.json')]
                done_ckpts[ckpt_name] = os.path.join(subdir, file)

    for ckpt_name in done_ckpts.keys():
        if ckpt_name in ckpts:
            del ckpts[ckpt_name]

    return ckpts


def _get_parser():
    parser = ArgumentParser(description='run_kp_eval.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)

    return parser


def summarize_scores(ckpt_name, dataset_score_dict):
    dataset_df_dict = {}
    for dataset, score_dict in dataset_score_dict.items():
        avg_dict = {}
        avg_dict['checkpoint_name'] = ckpt_name
        present_num = score_dict['present_num']
        absent_num = score_dict['absent_num']
        del score_dict['present_num'], score_dict['absent_num']

        for score_name, score_list in score_dict.items():
            if score_name.startswith('present'):
                tmp_scores = [score for score, num in zip(score_list, present_num) if num > 0]
                avg_dict[score_name] = np.average(tmp_scores)
            elif score_name.startswith('absent'):
                tmp_scores = [score for score, num in zip(score_list, absent_num) if num > 0]
                avg_dict[score_name] = np.average(tmp_scores)
            else:
                raise NotImplementedError

        summary_df = pd.DataFrame.from_dict(avg_dict, orient='index').transpose()
        dataset_df_dict[dataset] = summary_df

    return dataset_df_dict


def export_summary_to_csv(json_root_dir, output_csv_path):
    all_df_dict = {}

    for subdir, dirs, files in os.walk(json_root_dir):
        for file in files:
            if not file.endswith('.score.json'):
                continue

            ckpt_name = file[: file.find('.score.json')]
            # key is dataset name, value is a dict whose key is metric name and value is a list of floats
            score_dict = json.load(open(os.path.join(subdir, file), 'r'))
            # ignore scores where no tgts available and return the average
            ckpt_df_dict = summarize_scores(ckpt_name, score_dict)

            for dataset, score_df in ckpt_df_dict.items():
                if dataset in all_df_dict:
                    all_df_dict[dataset] = all_df_dict[dataset].append(score_df)
                else:
                    all_df_dict[dataset] = score_df

    for dataset, score_df in all_df_dict.items():
        score_df.to_csv(output_csv_path % dataset)


if __name__ == "__main__":
    parser = _get_parser()

    parser.add_argument('-ckpt_dir', type=str, required=True, help='Directory to all checkpoints')
    parser.add_argument('-output_dir', type=str, required=True, help='Directory to output results')
    parser.add_argument('-data_dir', type=str, required=True, help='Directory to datasets (ground-truth)')
    parser.add_argument('-testsets', nargs='+', type=str, default=["nus", "semeval"], help='Specify datasets to test on')
    # parser.add_argument('-testsets', nargs='+', type=str, default=["kp20k", "duc", "inspec", "krapivin", "nus", "semeval"], help='Specify datasets to test on')

    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logger = init_logger(opt.output_dir + 'autoeval_%s.log' % current_time)

    testset_path_dict = {}
    for testset in opt.testsets:
        src_shard = split_corpus(opt.data_dir + '/%s/%s_test.src' % (testset, testset), shard_size=-1)
        tgt_shard = split_corpus(opt.data_dir + '/%s/%s_test.tgt' % (testset, testset), shard_size=-1)
        src_shard, tgt_shard = list(zip(src_shard, tgt_shard))[0]
        testset_path_dict[testset] = (opt.data_dir + '/%s/%s_test.src' % (testset, testset),
                                      opt.data_dir + '/%s/%s_test.tgt' % (testset, testset),
                                      src_shard, tgt_shard)

    while True:
        new_ckpts = scan_new_checkpoints(opt.ckpt_dir, opt.output_dir)
        print(new_ckpts.items())
        print(sorted(new_ckpts.items(), key=lambda x:int(x[0][x[0].rfind('step_')+5:])))
        for ckpt_name, ckpt_path in sorted(new_ckpts.items(), key=lambda x:int(x[0][x[0].rfind('step_')+5:])):
            logger.info("Loading model from checkpoint: %s" % ckpt_path)
            setattr(opt, 'models', [ckpt_path])

            translator = build_translator(opt, report_score=opt.verbose, logger=logger)

            score_dicts = {}
            for dataname, dataset in testset_path_dict.items():
                src_path, tgt_path, src_shard, tgt_shard = dataset
                if not os.path.exists(os.path.join(opt.output_dir, 'pred', ckpt_name)):
                    os.makedirs(os.path.join(opt.output_dir, 'pred', ckpt_name))

                logger.info("Start translating %s." % dataname)
                pred_path = os.path.join(opt.output_dir, 'pred', ckpt_name, '%s.pred' % dataname)
                # set output_file here for each dataset
                translator.out_file = codecs.open(pred_path, 'w+', 'utf-8')
                _, _ = translator.translate(
                    src=src_shard,
                    tgt=tgt_shard,
                    src_dir=opt.src_dir,
                    batch_size=opt.batch_size,
                    attn_debug=opt.attn_debug,
                    opt=opt
                )
                logger.info("Start evaluating generated results of %s" % ckpt_name)
                score_dict = kp_evaluate.keyphrase_eval(src_path, tgt_path,
                                                        pred_path=pred_path, logger=logger, verbose=opt.verbose)
                score_dicts[dataname] = score_dict

            # print(score_dict)
            with open(os.path.join(opt.output_dir, ckpt_name+'.score.json'), 'w') as output_json:
                output_json.write(json.dumps(score_dicts))

        export_summary_to_csv(opt.output_dir, os.path.join(opt.output_dir, '%s_summary_%s.csv' % (current_time, '%s')))

        # scan again for every 5min
        sleep_time = 600
        logger.info('Sleep for %d sec' % sleep_time)
        time.sleep(sleep_time)
