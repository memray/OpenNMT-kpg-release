# -*- coding: utf-8 -*-
"""
Load averaged results from csv, containing scores of all ckpts.
For each exp group, return the best ckpt (ranked by valid performance).
"""
import shutil

import configargparse
import os

import pandas as pd

from kp_evaluate import gather_eval_results

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

train_test_mappings = {
    # 'kp20k': ['kp20k', 'kp20k_valid2k', 'inspec', 'krapivin', 'semeval', 'nus', 'duc'],
    'kp20k': ['kp20k', 'kp20k_valid2k', 'duc'],
    'openkp': ['openkp', 'openkp_valid2k', 'duc'],
    'kptimes': ['kptimes', 'kptimes_valid2k', 'jptimes', 'duc'],
    'stackex': ['stackex', 'stackex_valid2k', 'duc'],
}

train_dev_pairs = [
    ('kp20k', 'kp20k_valid2k'),
    ('openkp', 'openkp_valid2k'),
    ('kptimes', 'kptimes_valid2k'),
    ('stackex', 'stackex_valid2k'),
]

dev_test_pairs = [
    ('kp20k_valid2k', 'kp20k'),
    ('openkp_valid2k', 'openkp'),
    ('kptimes_valid2k', 'kptimes'),
    ('kptimes_valid2k', 'jptimes'),
    ('kptimes_valid2k', 'duc'),
    ('stackex_valid2k', 'stackex'),
]

def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-exp_base_dir', type=str, required=True, help='source ckpt/pred/eval files.')
    parser.add_argument('-export_base_dir', type=str, required=True, help='The best ckpt/pred/eval files will be copied to this place.')
    parser.add_argument('-decoding_method', type=str, required=True, help='Filter by decoding_method, since there exists results by multiple decoding settings from the same ckpt, like beamsearch-width_50-maxlen_40.')
    parser.add_argument('-accept_unfinished', action='store_true', help='if inference job is not all done, ignore this group')
    opt = parser.parse_args()

    for exp_name in os.listdir(opt.exp_base_dir):
        exp_dir = os.path.join(opt.exp_base_dir, exp_name)
        ckpt_dir = os.path.join(exp_dir, 'ckpts')
        if not os.path.exists(ckpt_dir): continue
        ckpts = [ckpt_file for ckpt_file in os.listdir(ckpt_dir) if ckpt_file.endswith('0.pt')]
        pred_dir = os.path.join(exp_dir, 'outputs', opt.decoding_method, 'pred')

        print('*' * 50)
        print('EXP name: %s' % exp_name)
        print('#ckpts=%d' % len(ckpts))

        export_ckpt_dir = os.path.join(opt.export_base_dir, exp_name, 'ckpts')
        export_pred_dir = os.path.join(opt.export_base_dir, exp_name, 'outputs', opt.decoding_method, 'pred')
        # if os.path.exists(export_pred_dir) and len(os.listdir(export_pred_dir)) > 0:
        #     print('Skip: already found %d files in export dir: \n\t\t\t%s' % (len(os.listdir(export_pred_dir)), export_pred_dir))
        #     continue

        train_name, dev_name = None, None
        for train, dev in train_dev_pairs:
            if train in exp_name:
                train_name = train
                dev_name = dev
                break

        if train_name is None:
            print('Trainset not found. Not a common experiment name? %s' % exp_name)
            continue

        if not os.path.exists(pred_dir):
            continue

        pred_files = [pred_file for pred_file in os.listdir(pred_dir) if pred_file.endswith('.pred')]
        dataset_scores_dict = gather_eval_results(pred_dir, report_csv_dir=None, tokenizer='split_nopunc')
        # for dataset_name, score_df in dataset_scores_dict.items():
        for dataset_name in train_test_mappings[train_name]:
            dataset_split_name = dataset_name + '_test'
            score_df = dataset_scores_dict[dataset_split_name] if dataset_split_name in dataset_scores_dict else []
            _pred_files = [filename for filename in pred_files if dataset_split_name in filename]
            print('\t#pred %s=%d' % (dataset_name, len(_pred_files)))
            print('\t#eval %s=%d' % (dataset_name, len(score_df)))

        pred_files = [filename for filename in pred_files if dev_name+'_test' in filename]
        dev_df = dataset_scores_dict[dev_name+'_test'] if dev_name+'_test' in dataset_scores_dict else None
        if not opt.accept_unfinished and (dev_df is None or len(dev_df) < len(pred_files) or len(pred_files) < len(ckpts)):
            print('Inference on devset (%s) has not accomplished: pred=%d/%d, eval=%d/%d '
                  % (dev_name,
                     len(pred_files), len(ckpts),
                     len(dev_df) if dev_df is not None else 0, len(pred_files)))
            continue

        if dev_df is None or len(dev_df) == 0:
            print('No inference found on devset (%s)')
            continue

        # pick up the best ckpt by dev score
        anchor_metric_name = 'all_exact_f_score@k'
        dev_df = dev_df.sort_values(by=anchor_metric_name, ascending=False)
        dev_row = dev_df.iloc[0].to_frame().transpose()

        print('best dev score: %s=%.4f' % (anchor_metric_name, dev_row[anchor_metric_name]))
        best_step = dev_row.step.item()
        best_ckpt_name_prefix = 'checkpoint_step_%s-' % best_step
        best_ckpt_filename = 'checkpoint_step_%s.pt' % best_step

        print('-' * 20)

        if not os.path.exists(export_ckpt_dir): os.makedirs(export_ckpt_dir)
        if not os.path.exists(export_pred_dir): os.makedirs(export_pred_dir)

        src_ckpt_path = os.path.join(ckpt_dir, best_ckpt_filename)
        if best_ckpt_filename in ckpts and os.path.exists(src_ckpt_path):
            print('Copy checkpoint file: %s' % best_ckpt_filename)
            tgt_ckpt_path = os.path.join(export_ckpt_dir, best_ckpt_filename)
            if os.path.exists(tgt_ckpt_path):
                print('Checkpoint file exists, skip: %s' % tgt_ckpt_path)
            else:
                shutil.copyfile(src_ckpt_path, tgt_ckpt_path)
        else:
            print('Checkpoint file not found: %s, path=%s' % (best_ckpt_filename, src_ckpt_path))

        for predeval_file in os.listdir(pred_dir):
            if predeval_file.startswith(best_ckpt_name_prefix):
                print('Copy pred/eval file: %s' % predeval_file)
                src_pred_path = os.path.join(pred_dir, predeval_file)
                tgt_pred_path = os.path.join(export_pred_dir, predeval_file)
                shutil.copyfile(src_pred_path, tgt_pred_path)

if __name__ == '__main__':
    main()