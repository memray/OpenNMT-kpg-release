import os

from tools.files2rouge import files2rouge

hyp_path = '/export/share/rmeng/output/roberta-base/exps/bart_released_cnndm/bart-released-cnndm/preds/beam_size5-min_length20-max_length140-stepwise_penaltyfalse-length_penaltynone-alpha2.0-coverage_penaltynone-beta0.0-block_ngram_repeat3/model_step_0.cnndm.test.summary.files2rouge.eval.hyp'
ref_path = '/export/share/rmeng/output/roberta-base/exps/bart_released_cnndm/bart-released-cnndm/preds/beam_size5-min_length20-max_length140-stepwise_penaltyfalse-length_penaltynone-alpha2.0-coverage_penaltynone-beta0.0-block_ngram_repeat3/model_step_0.cnndm.test.summary.files2rouge.eval.ref'

rouge_setting_path = '/export/share/rmeng/project/OpenNMT-summary/tools/files2rouge/settings.json'
files2rouge.run(hyp_path, ref_path, path=rouge_setting_path)
