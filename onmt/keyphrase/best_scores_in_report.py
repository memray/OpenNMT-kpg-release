# -*- coding: utf-8 -*-
"""
Load averaged results from csv, containing scores of all ckpts.
For each exp group, return the best ckpt (ranked by valid performance).
"""
import shutil

import configargparse
import os

import pandas as pd



__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

dev_test_pairs = [
    ('kp20k_valid2k', 'kp20k'),
    # ('kp20k_valid2k', 'duc'),
    ('openkp_valid2k', 'openkp'),
    ('kptimes_valid2k', 'kptimes'),
    # ('kptimes_valid2k', 'jptimes'),
    # ('kptimes_valid2k', 'duc'),
    ('stackex_valid2k', 'stackex'),
    # ('kp20k_valid2k', 'inspec'),
    # ('kp20k_valid2k', 'krapivin'),
    # ('kp20k_valid2k', 'nus'),
    # ('kp20k_valid2k', 'semeval'),
]

def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-report_dir', type=str, required=True, help='Directory to all report csv files.')
    parser.add_argument('-pred_name', type=str, required=False, help='Filter by pred_name, since there exists results by multiple decoding settings from the same ckpt.')
    parser.add_argument('-export_dir', type=str, required=False, default=None, help='If set, the best pred/eval files will be copied to this place.')
    parser.add_argument('-report_selfbest', action='store_true', help='')
    parser.add_argument('-report_lastckpt', action='store_true', help='')
    opt = parser.parse_args()

    kp_df = None
    for f in os.listdir(opt.report_dir):
        if not f.endswith('.csv'): continue
        print(f)
        df = pd.read_csv(os.path.join(opt.report_dir, f))
        kp_df = df if kp_df is None else pd.concat([kp_df, df], sort=True)

    # rearrange cols since paths take a lot of space
    cols = df.columns.tolist()
    path_cols = [c for c in cols if c.endswith('_path')]
    not_path_cols = [c for c in cols if not c.endswith('_path')]
    kp_df = kp_df[not_path_cols + path_cols]

    if opt.pred_name is None:
        if len(kp_df.pred_name.unique()) > 1:
            print('Found multiple decoding settings, please set opt.pred_name to avoid mixed results')
            print(kp_df.pred_name.unique().tolist())
            raise Exception()
    else:
        kp_df = kp_df.loc[kp_df.pred_name == opt.pred_name]

    # kp_df = kp_df.loc[kp_df.exp_name.str.contains("PT_step200k")]

    print(len(kp_df))
    # print(kp_df.columns)
    for exp_name in kp_df.exp_name.unique():
        print(exp_name, len(kp_df.loc[kp_df.exp_name == exp_name]))
    exp_names = kp_df.exp_name.unique()

    anchor_metric_name = 'all_exact_f_score@k'
    # anchor_metric_name = 'present_exact_f_score@k'

    for dev_test_pair in dev_test_pairs:
        # for transfer results
        dev_name = dev_test_pair[0] + '_test'
        test_name = dev_test_pair[1] + '_test'
        # for empirical results
        # dev_name = dev_test_pair[0]
        # test_name = dev_test_pair[1]

        devbest_dev_rows, devbest_test_rows = None, None
        selfbest_dev_rows, selfbest_test_rows = None, None
        for exp_name in exp_names:
            exp_df = kp_df.loc[kp_df.exp_name == exp_name]
            dev_df = exp_df.loc[exp_df.test_dataset == dev_name]
            test_df = exp_df.loc[exp_df.test_dataset == test_name]
            if opt.report_lastckpt:
                dev_df = dev_df.sort_values(by='step', ascending=False)
                test_df = test_df.sort_values(by='step', ascending=False)
            else:
                dev_df = dev_df.sort_values(by=anchor_metric_name, ascending=False)
                test_df = test_df.sort_values(by=anchor_metric_name, ascending=False)

            if len(dev_df) == 0: continue
            dev_row = dev_df.iloc[0].to_frame().transpose()
            selfbest_dev_row = dev_row
            selfbest_dev_rows = dev_row if selfbest_dev_rows is None else pd.concat([selfbest_dev_rows, dev_row])

            if len(test_df) > 0:
                test_row = test_df.iloc[0].to_frame().transpose()
                selfbest_test_rows = test_row if selfbest_test_rows is None else pd.concat([selfbest_test_rows, test_row])

            test_row = None
            for idx, dev_row in dev_df.iterrows():
                best_step = dev_row.step
                test_row = test_df.loc[test_df.step == best_step]
                if len(test_row) == 1:
                    dev_row = dev_row.to_frame().transpose()
                    test_row = test_row.iloc[0].to_frame().transpose()
                    break
                elif len(test_row) > 1:
                    print('Found multiple rows (%d rows): exp=%s, data=%s' % (
                        len(test_row), exp_name, str(dev_test_pair)))
                    raise ValueError()
                # elif len(test_row) == 0:
                #     print('Corresponding test row not found: exp=%s, data=%s' % (exp_name, str(dev_test_pair)))
                # else:
                #     print('what?')

            if test_row is not None and len(test_row) > 0:
                devbest_dev_rows = dev_row if devbest_dev_rows is None else pd.concat([devbest_dev_rows, dev_row])
                devbest_test_rows = test_row if devbest_test_rows is None else pd.concat([devbest_test_rows, test_row])

                # move the best dev pred/eval to the specified place
                if opt.export_dir is not None:
                    with pd.option_context('display.max_colwidth', 9999999):
                        pred_file_path = test_row.pred_file_path.to_string(index=False)
                        eval_file_path = test_row.eval_file_path.to_string(index=False)
                        target_dir = os.path.join(opt.export_dir, 'dev_best', exp_name, '')
                        exp_name = test_row.exp_name.to_string(index=False)
                        pred_file_name = pred_file_path[pred_file_path.rfind('/') + 1: ]
                        eval_file_name = eval_file_path[pred_file_path.rfind('/') + 1:]
                        target_pred_file = '-'.join([exp_name, pred_file_name])
                        target_eval_file = '-'.join([exp_name, eval_file_name])

                    if not os.path.exists(target_dir): os.makedirs(target_dir)
                    shutil.copyfile(pred_file_path, target_dir + target_pred_file)
                    shutil.copyfile(eval_file_path, target_dir + target_eval_file)
            else:
                devbest_dev_rows = selfbest_dev_row if devbest_dev_rows is None else pd.concat([devbest_dev_rows, selfbest_dev_row])
                print('Cannot find valid dev-best rows: exp=%s, data=%s, dev_df.size=%d, test_df.size=%d'
                      % (exp_name, str(dev_test_pair), len(dev_df), len(test_df)))
            # print(exp_name)
            # print('len(devbest_dev_rows)=', len(devbest_dev_rows))
            # print('len(selfbest_dev_rows)=', len(selfbest_dev_rows))

        print('Dev best - ' + dev_name)
        print(devbest_dev_rows.sort_values(by=anchor_metric_name, ascending=False).to_csv()) if devbest_dev_rows is not None else print('Empty')
        print('=' * 50)
        print('Dev best - ' + test_name)
        print(devbest_test_rows.sort_values(by=anchor_metric_name, ascending=False).to_csv()) if devbest_test_rows is not None else print('Empty')
        print('=' * 30)

        if opt.report_selfbest:
            print('Self best - ' + dev_name)
            print(selfbest_dev_rows.sort_values(by=anchor_metric_name, ascending=False).to_csv()) if selfbest_dev_rows is not None else print('Empty')
            print('=' * 50)
            print('Self best - ' + test_name)
            print(selfbest_test_rows.sort_values(by=anchor_metric_name, ascending=False).to_csv()) if selfbest_test_rows is not None else print('Empty')

        print('*' * 50)


if __name__ == '__main__':
    main()