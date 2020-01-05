import json
import tqdm
import numpy as np

pred_path = "/Users/memray/project/kp/OpenNMT-kpg/output/aaai20/catseq_pred/kp20k.pred"
pred_path = "/Users/memray/project/kp/OpenNMT-kpg/output/aaai20/catseqd_pred/kp20k.pred"

keys = ['gold_num',
        'unique_pred_num', 'dup_pred_num', 'pred_sents_local_count', 'topseq_pred_num',
        'beam_num', 'beamstep_num']
num_doc = 0
stat_dict = {k: [] for k in keys}

for l in tqdm.tqdm(open(pred_path, 'r')):
    pred_dict = json.loads(l)
    #     print(pred_dict.keys())
    print(pred_dict['topseq_pred_sents'])  # top beam, a sequence of words
    print(pred_dict['topseq_preds'])  # a sequence of indices

    print(pred_dict['pred_sents'])  # unique phrases
    print(pred_dict['ori_pred_sents'])  # beams, each is a list of words, seperated by <sep>
    #     print(pred_dict['ori_preds'])

    print(pred_dict['unique_pred_num'])
    if num_doc > 10:
        break

    num_doc += 1
    stat_dict['gold_num'].append(len(pred_dict['gold_sent']))
    stat_dict['unique_pred_num'].append(pred_dict['unique_pred_num'])
    stat_dict['dup_pred_num'].append(pred_dict['dup_pred_num'])
    stat_dict['pred_sents_local_count'].append(len(pred_dict['pred_sents']))
    stat_dict['beam_num'].append(pred_dict['beam_num'])
    stat_dict['beamstep_num'].append(pred_dict['beamstep_num'])
    stat_dict['topseq_pred_num'].append(len(pred_dict['topseq_pred_num']))

    print('#(doc)=%d' % num_doc)
    for k, v in stat_dict.items():
        print('avg(%s) = %d/%d = %f' % (k, np.sum(v), num_doc, np.mean(v)))
