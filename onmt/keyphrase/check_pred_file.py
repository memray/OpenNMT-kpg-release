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
    pred_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-v3/meng17-one2seq-fullbeam/meng17-one2seq-beam50-maxlen40/pred/kpgen-meng17-kp20k+MagKP_Nlarge-verbatim_append-transformer-L6H8-BS4096-LR0.05-L6-H8-Dim512-Emb512-Dropout0.1-Copytrue-Covtrue_step_120000/kp20k_valid2k.pred'

    # ensure the pred is complete
    with open(pred_path, 'r') as pred_file:
        for lid, line in enumerate(pred_file):
            try:
                pred_dict = json.loads(line)
            except:
                print("Error occurs while loading line %d" % (lid))
                print(line)
                continue
            # for k,v in pred_dict.items():
            #     print('%s' % k)
