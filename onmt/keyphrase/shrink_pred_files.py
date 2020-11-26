# -*- coding: utf-8 -*-
"""
Some pred files use up too much space, e.g. /zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/pred/kp20k-meng17-verbatim_prepend-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_95000/kp20k.pred is 8.3GB, beam=10 size=2.0GB.

So this
"""
import json
import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    root_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-v3/meng17-one2seq-fullbeam/'
    # root_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-v2/meng17-one2seq-fullbeam/'
    # root_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/'
    # root_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-v3/meng17-one2one-fullbeam/'
    # root_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/'

    # root_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/order_matters/transformer/meng17-one2seq-beam50-maxlen40/'
    dataset_line_counts = {'kp20k': 19987,
                     # 'kp20k_valid2k': 2000,
                     # 'inspec': 500,
                     # 'krapivin': 460,
                     # 'nus': 211,
                     # 'semeval': 100,
                     # 'duc': 308,
                     # 'stackexchange': 16000,
                     }

    total_size_shrinked = 0
    for root, dirs, files in os.walk(root_path, topdown=True):
        for filename in files:
            # print()
            # print('-=' * 50)
            # print(filename)
            # print('-=' * 50)

            if not filename.endswith('.pred'):
                continue
            dataset_name = filename[:-5]
            if dataset_name not in dataset_line_counts:
                # print(dataset_name + ' is not within shrinking list, skip! ')
                continue

            pred_path = os.path.join(root, filename)
            print('Shrinking file: [%s] %s' % (dataset_name, pred_path))
            ori_size = os.stat(pred_path).st_size // 1024 // 1024
            print('\t file size = %d MB' % (ori_size))

            # ensure the pred is complete
            with open(pred_path, 'r') as pred_file:
                lines = [l if lid==0 else '' for lid, l in enumerate(pred_file)]
                if len(lines) != dataset_line_counts[dataset_name]:
                    print('Prediction ongoing, skip!')
                    continue

                pred_dict = json.loads(lines[0])
                # indicating it's already shrinked, skip
                if pred_dict['attns'] == None and pred_dict['dup_pred_tuples'] == None:
                    print('This pred file has been shrinked, skip!')
                    continue

            tmp_pred_path = pred_path + '.tmp'
            tmp_pred_file = open(tmp_pred_path, 'w')
            with open(pred_path, 'r') as pred_file:
                for lid, line in enumerate(pred_file):
                    try:
                        pred_dict = json.loads(line)
                    except:
                        tmp_pred_file.write(line.strip() + '\n')
                        print("Error occurs while loading line %d in %s" % (lid, pred_path))
                        continue
                    # for k,v in pred_dict.items():
                    #     print('%s' % k)

                    pred_dict['attns'] = None
                    pred_dict['ori_pred_sents'] = None
                    pred_dict['ori_pred_scores'] = None
                    pred_dict['ori_preds'] = None
                    pred_dict['dup_pred_tuples'] = None
                    tmp_pred_file.write(json.dumps(pred_dict)+'\n')

            # tmp_pred_file.close()
            print('Dumped to: ' + pred_path + '.tmp')
            new_size = os.stat(tmp_pred_path).st_size // 1024 // 1024
            print('\t new file size = %d MB' % (new_size))
            print('\t shrinked size = %d MB' % (ori_size-new_size))

            total_size_shrinked += (ori_size - new_size)

            # replace the original file to release space
            os.remove(pred_path)
            os.rename(tmp_pred_path, pred_path)

    print('Total shrinked size = %d MB' % (total_size_shrinked))

