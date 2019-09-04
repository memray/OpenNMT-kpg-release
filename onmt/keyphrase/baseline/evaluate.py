# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

from kp_evaluate import init_opt

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    opt = init_opt()
    score_dicts = {}

    for ckpt_name in os.listdir(opt.pred_dir):

        for dataname in opt.testsets:
            src_path = os.path.join(opt.data, dataname, "%s_test.src" % dataname)
            tgt_path = os.path.join(opt.data, dataname, "%s_test.tgt" % dataname)
            pred_path = os.path.join(opt.pred_dir, ckpt_name, "%s.pred" % dataname)
