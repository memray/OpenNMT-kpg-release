import codecs
import json
import re
import numpy as np

import tqdm
from onmt.newssum import docutils
from tools.files2rouge import files2rouge
from tools.test_rouge import evaluate_chunk_coverage, rouge_results_to_str, test_rouge_perl, test_rouge_py

rouge_impl = 'py'
stanford_token = True
hyp = '/export/share/rmeng/tools/torchhub/bart.large.cnn/bart.large.cnndm.test.hyp'
ref = '/export/share/rmeng/output/word/tokenized/cnndm/test.sorted.jsonl'

hyp_path = '/export/share/rmeng/tools/torchhub/bart.large.cnn/bart.large.cnndm.files2rouge.eval.hyp'
ref_path = '/export/share/rmeng/tools/torchhub/bart.large.cnn/bart.large.cnndm.files2rouge.eval.ref'

hyp_file = codecs.open(hyp, encoding="utf-8")
ref_file = codecs.open(ref, encoding="utf-8")

origin_candidates = [line.strip() for line in tqdm.tqdm(hyp_file, desc='Loading generated summaries')]
origin_references = [line.strip() for line in tqdm.tqdm(ref_file, desc='Loading ground-truth summaries')]

candidates = origin_candidates
references = origin_references

# post-process references (ground-truth summaries)
print('Post-processing references (ground-truth summaries)')
gt_dicts = [json.loads(s) for s in references]
references = [s['word']['token']['summary'] for s in gt_dicts]
references = [' '.join(l) for l in references]

# remove special tokens like '[SEP_SUM]', '\n', '</s>'
candidates = [re.sub(r'\[.*?\]', '', c) for c in candidates]
candidates = [c.replace('\n', ' ').replace('[SEP_SUM]', ' ').replace('</s>', ' ') for c in candidates]
candidates = [re.sub(r'-(\s+)-', '--', c.replace('-', ' - ')) for c in candidates]
references = [r.replace('\n', ' ').replace('[SEP_SUM]', ' ') for r in references]
references = [re.sub(r'-(\s+)-', '--', r.replace('-', ' - ')) for r in references]

# whether to apply another round of Stanford Tokenization
print("Checking sequence length...")
if stanford_token:
    print("Stanford Tokenizing...")
    candidates = [docutils.word_tokenize(c, model="stanfordnlp") for c in
                  tqdm.tqdm(candidates, desc='Stanford tokenizing generated summaries')]
    references = [docutils.word_tokenize(r, model="stanfordnlp") for r in
                  tqdm.tqdm(references, desc='Stanford tokenizing ground-truth summaries')]
    leng_stat = {'ref_avg_len': np.mean([len(r) for r in references]),
                 'hyp_avg_len': np.mean([len(c) for c in candidates])}
else:
    candidates = [c.split() for c in candidates]
    references = [r.split() for r in references]
    leng_stat = {'ref_avg_len': np.mean([len(r) for r in references]),
                 'hyp_avg_len': np.mean([len(c) for c in candidates])}
print(leng_stat)
candidates = [' '.join(c) for c in candidates]
references = [' '.join(r) for r in references]

print("Lowercasing the input text...")
candidates = [c.lower() for c in candidates]
references = [r.lower() for r in references]

# add NER and NP coverage
ner_coverage = evaluate_chunk_coverage(pred_strs=candidates, gt_dicts=gt_dicts, key='ner')
np_coverage = evaluate_chunk_coverage(pred_strs=candidates, gt_dicts=gt_dicts, key='noun_phrase')
print(ner_coverage)
print(np_coverage)

print("Start calculating rouge with %s..." % rouge_impl)
if rouge_impl == 'perl':
    # pyrouge is pathetically slow once number of docs is large (on newsroom), due to three file exporting steps:
    #   (1) temp files 1 (~17min); (2)  temp files 2 to /tmp (~2min);
    #   (3) generating config file (at line 504 __get_model_filenames_for_id() very very slow, >>30min),
    #       bcuz it's doing n^2 file scanning. Change to no list_dir() version.
    results_dict = test_rouge_perl(candidates, references,
                                   rouge_path='/export/share/rmeng/project/OpenNMT-summary/tools/ROUGE-1.5.5/')
elif rouge_impl == 'py':
    # a pure python implementation
    results_dict = test_rouge_py(candidates, references)
elif rouge_impl == 'files2rouge':
    # dump output to files and run files2rouge
    print("Dumping candidates and references to: %s \nand: %s" % (hyp_path, ref_path))
    with open(hyp_path, 'w') as h_file:
        for cl in candidates:
            h_file.write(cl + '\n')
    with open(ref_path, 'w') as r_file:
        for rl in references:
            r_file.write(rl + '\n')
    results_dict = files2rouge.run(hyp_path, ref_path,
                                   path='/export/share/rmeng/project/OpenNMT-summary/tools/files2rouge/settings.json')

results_dict.update(leng_stat)
if 'ner_coverage' in locals():
    results_dict.update(ner_coverage)
if 'np_coverage' in locals():
    results_dict.update(np_coverage)

print(rouge_results_to_str(results_dict))

print(results_dict)

"""
rouge_impl = 'files2rouge'
stanford_token = True
lower = True
{'ref_avg_len': 60.00661444734552, 'hyp_avg_len': 74.15596170583116}
---------------------------------------------
1 ROUGE-1 Average_R: 0.51361 (95%-conf.int. 0.51104 - 0.51626)
1 ROUGE-1 Average_P: 0.40549 (95%-conf.int. 0.40308 - 0.40808)
1 ROUGE-1 Average_F: 0.44235 (95%-conf.int. 0.44011 - 0.44458)
---------------------------------------------
1 ROUGE-2 Average_R: 0.24611 (95%-conf.int. 0.24338 - 0.24883)
1 ROUGE-2 Average_P: 0.19476 (95%-conf.int. 0.19245 - 0.19701)
1 ROUGE-2 Average_F: 0.21209 (95%-conf.int. 0.20971 - 0.21441)
---------------------------------------------
1 ROUGE-3 Average_R: 0.14789 (95%-conf.int. 0.14524 - 0.15057)
1 ROUGE-3 Average_P: 0.11727 (95%-conf.int. 0.11499 - 0.11937)
1 ROUGE-3 Average_F: 0.12748 (95%-conf.int. 0.12518 - 0.12969)
---------------------------------------------
1 ROUGE-L Average_R: 0.47636 (95%-conf.int. 0.47370 - 0.47896)
1 ROUGE-L Average_P: 0.37638 (95%-conf.int. 0.37393 - 0.37880)
1 ROUGE-L Average_F: 0.41045 (95%-conf.int. 0.40809 - 0.41264)

Elapsed time: 220.838 seconds
rouge_1_recall: 0.51361
rouge_1_precision: 0.40549
rouge_1_f_score: 0.44235
rouge_2_recall: 0.24611
rouge_2_precision: 0.19476
rouge_2_f_score: 0.21209
rouge_3_recall: 0.14789
rouge_3_precision: 0.11727
rouge_3_f_score: 0.12748
rouge_l_recall: 0.47636
rouge_l_precision: 0.37638
rouge_l_f_score: 0.41045

ref_avg_len: 60.00661444734552
hyp_avg_len: 74.15596170583116

ner_corr_num: 3.5452567449956485
ner_all_num: 5.872149695387293
ner_cov: 0.6122657040375573

noun_phrase_corr_num: 6.301044386422976
noun_phrase_all_num: 13.929503916449086
noun_phrase_cov: 0.4558219337567122
"""

"""
rouge_impl = 'py'
stanford_token = True
lower = True
{'ref_avg_len': 60.00661444734552, 'hyp_avg_len': 74.15596170583116}
{'ner_corr_num': 3.5452567449956485, 'ner_all_num': 5.872149695387293, 'ner_cov': 0.6122657040375573}
{'noun_phrase_corr_num': 6.301044386422976, 'noun_phrase_all_num': 13.929503916449086, 'noun_phrase_cov': 0.4558219337567122}
rouge_1_recall: 0.51702
rouge_1_precision: 0.40819
rouge_1_f_score: 0.44528

rouge_2_recall: 0.24702
rouge_2_precision: 0.19546
rouge_2_f_score: 0.21287

rouge_3_recall: 0.14847
rouge_3_precision: 0.11771
rouge_3_f_score: 0.12797

rouge_l_recall: 0.35933
rouge_l_precision: 0.28188
rouge_l_f_score: 0.30834

rouge_w_recall: 0.13187
rouge_w_precision: 0.22244
rouge_w_f_score: 0.16036

ref_avg_len: 60.00661444734552
hyp_avg_len: 74.15596170583116

ner_corr_num: 3.5452567449956485
ner_all_num: 5.872149695387293
ner_cov: 0.6122657040375573

noun_phrase_corr_num: 6.301044386422976
noun_phrase_all_num: 13.929503916449086
noun_phrase_cov: 0.4558219337567122
"""


"""
rouge_impl = 'files2rouge'
stanford_token = False
lower = True
{'ref_avg_len': 59.97023498694517, 'hyp_avg_len': 66.10382941688425}
{'ner_corr_num': 3.529068755439513, 'ner_all_num': 5.872149695387293, 'ner_cov': 0.6092303707411274}
{'noun_phrase_corr_num': 6.216100957354221, 'noun_phrase_all_num': 13.929503916449086, 'noun_phrase_cov': 0.4495942981856436}
---------------------------------------------
1 ROUGE-1 Average_R: 0.51246 (95%-conf.int. 0.50989 - 0.51517)
1 ROUGE-1 Average_P: 0.40520 (95%-conf.int. 0.40281 - 0.40780)
1 ROUGE-1 Average_F: 0.44174 (95%-conf.int. 0.43946 - 0.44392)
---------------------------------------------
1 ROUGE-2 Average_R: 0.24477 (95%-conf.int. 0.24203 - 0.24753)
1 ROUGE-2 Average_P: 0.19400 (95%-conf.int. 0.19169 - 0.19622)
1 ROUGE-2 Average_F: 0.21112 (95%-conf.int. 0.20877 - 0.21343)
---------------------------------------------
1 ROUGE-3 Average_R: 0.14664 (95%-conf.int. 0.14402 - 0.14926)
1 ROUGE-3 Average_P: 0.11647 (95%-conf.int. 0.11419 - 0.11856)
1 ROUGE-3 Average_F: 0.12652 (95%-conf.int. 0.12422 - 0.12871)
---------------------------------------------
1 ROUGE-L Average_R: 0.43126 (95%-conf.int. 0.42863 - 0.43390)
1 ROUGE-L Average_P: 0.34166 (95%-conf.int. 0.33930 - 0.34404)
1 ROUGE-L Average_F: 0.37212 (95%-conf.int. 0.36998 - 0.37432)

Elapsed time: 218.586 seconds
rouge_1_recall: 0.51246
rouge_1_precision: 0.4052
rouge_1_f_score: 0.44174
rouge_2_recall: 0.24477
rouge_2_precision: 0.194
rouge_2_f_score: 0.21112
rouge_3_recall: 0.14664
rouge_3_precision: 0.11647
rouge_3_f_score: 0.12652
rouge_l_recall: 0.43126
rouge_l_precision: 0.34166
rouge_l_f_score: 0.37212

ref_avg_len: 59.97023498694517
hyp_avg_len: 66.10382941688425

ner_corr_num: 3.529068755439513
ner_all_num: 5.872149695387293
ner_cov: 0.6092303707411274

noun_phrase_corr_num: 6.216100957354221
noun_phrase_all_num: 13.929503916449086
noun_phrase_cov: 0.4495942981856436
"""