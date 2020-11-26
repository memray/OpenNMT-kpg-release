# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

import kp_evaluate

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    eval_dir = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-v2/'
    dataset_scores_dict = kp_evaluate.gather_eval_results(eval_root_dir=eval_dir)

    print(dataset_scores_dict)
