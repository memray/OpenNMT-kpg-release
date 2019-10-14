# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

from kp_evaluate import init_opt, keyphrase_eval, kp_results_to_str, export_summary_to_csv
from onmt.utils.logging import init_logger

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    opt = init_opt()
    score_dicts = {}

    for model_name in os.listdir(opt.pred_dir):
        pred_dir = os.path.join(opt.pred_dir, model_name, "pred")
        if not os.path.exists(pred_dir):
            continue

        for dataset in opt.testsets:
            logger = init_logger(os.path.join(opt.output_dir, model_name, "kp_evaluate.log"))
            logger.info("Evaluating model %s on %s" % (model_name, dataset))

            src_path = os.path.join(opt.data, dataset, "%s_test.src" % dataset)
            tgt_path = os.path.join(opt.data, dataset, "%s_test.tgt" % dataset)
            pred_path = os.path.join(opt.pred_dir, model_name, "pred", "%s" % dataset)

            if not os.path.exists(os.path.join(opt.output_dir, 'eval')):
                os.makedirs(os.path.join(opt.output_dir, 'eval'))

            score_path = os.path.join(opt.output_dir, 'eval', model_name + '-%s.json' % dataset)

            if not os.path.exists(score_path):
                score_dict = keyphrase_eval(src_path=src_path,
                                            tgt_path=tgt_path,
                                            pred_path=pred_path,
                                            unk_token = '<unk>',
                                            verbose = opt.verbose,
                                            logger = logger,
                                            eval_topbeam=opt.topbeam,
                                            model_name=model_name
                                            )
                if score_dict is not None:
                    logger.info(kp_results_to_str(score_dict))
                    with open(score_path, 'w') as output_json:
                        output_json.write(json.dumps(score_dict))
                    score_dicts[dataset] = score_dict
            else:
                logger.error("Skip evaluating as previous eval result exists")

        export_summary_to_csv(json_root_dir=os.path.join(opt.output_dir),
                              report_csv_path=os.path.join(opt.output_dir, 'summary_%s.csv' % ('%s')))

        logger.info("Done!")