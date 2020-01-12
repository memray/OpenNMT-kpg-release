import hashlib
import json

import tqdm

import onmt.newssum.docutils as docutils


def load_groundtruth_data(groundtruth_jsonl_path):
    all_gt_ex_dicts = []
    for line_id, jsonl in enumerate(open(groundtruth_jsonl_path, 'r')):
        ex = json.loads(jsonl)
        all_gt_ex_dicts.append(ex)
    print('Loaded %s groundtruth data examples' % len(all_gt_ex_dicts))

    # fix the bug that doc['oracle']['bottomup']['oracle_text'] is set as sentence oracle by mistake
    for ex in tqdm.tqdm(all_gt_ex_dicts):
        title_tokens = ex['word']['token']['title']
        sentences_tokens = ex['word']['token']['sents']
        title_sents_tokens = [title_tokens] + [['\n']] + sentences_tokens
        summary_tokens = ex['word']['token']['summary']
        #     print('[SUMMARY]:' + str(summary_tokens))

        oracle_bottomup_mask = ex['word']['oracle']['bottomup']['mask']
        oracle_bottomup_text = docutils.mask_to_text(title_sents_tokens, oracle_bottomup_mask)
        ex['word']['oracle']['bottomup']['oracle_text'] = oracle_bottomup_text
        #     print('[BOTTOMUP]:' + str(oracle_bottomup_text))
        #     print('[BOTTOMUP]:' + str(ex['word']['oracle']['bottomup']['oracle_rouge']))

        oracle_sent_ids = ex['word']['oracle']['sentence']['target']
        oracle_sent_text = ' '.join([w for sid in oracle_sent_ids for w in title_sents_tokens[sid]])
        oracle_sent_text = oracle_sent_text.replace('\n', ' ')
        ex['word']['oracle']['sentence']['oracle_text'] = oracle_sent_text
    #     print('[SENT=%d]: %s' % (len(oracle_sent_ids), str(oracle_sent_text)))
    #     print('[SENT]:' + str(ex['word']['oracle']['sentence']['oracle_rouge']))

    return all_gt_ex_dicts

def hashhex(s):
    '''Returns a heximal formated SHA1 hash of the input string.'''
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

groundtruth_jsonl_path = "/export/share/rmeng/output/word/tokenized/cnndm/test.jsonl"
all_gt_ex_dicts = load_groundtruth_data(groundtruth_jsonl_path)
print('Loaded %s groundtruth data examples' % len(all_gt_ex_dicts))

# load urls from '/export/share/rmeng/data/raw/cnndm/original/cnn/wayback_test_urls.txt' and hash it with hashhex()
cnn_url_path = '/export/share/rmeng/data/raw/cnndm/original/cnn/wayback_test_urls.txt'
dm_url_path = '/export/share/rmeng/data/raw/cnndm/original/dailymail/wayback_test_urls.txt'

urls = [l.strip() for l in open(cnn_url_path, 'r').readlines()]
cnn_docids = ['cnn-test-' + hashhex(url) for url in urls]

urls = [l.strip() for l in open(dm_url_path, 'r').readlines()]
dm_docids = ['dailymail-test-' + hashhex(url) for url in urls]

doc_ids = cnn_docids + dm_docids
print('#(doc_id)=%d' % len(doc_ids))

srcidx2id = {}
id2ex_dict = {}
for srcidx, ex in enumerate(all_gt_ex_dicts):
    id2ex_dict[ex['id']] = ex

id2tgtidx = {}
gt_ex_dicts = []
# resort all_gt_ex_dicts following the order of mappings
for tgtidx, doc_id in enumerate(doc_ids):
    ex = id2ex_dict[doc_id]
    gt_ex_dicts.append(ex)
    id2tgtidx[doc_id] = tgtidx

print('Resorted %s groundtruth data examples' % len(gt_ex_dicts))

# our order -> original order
map_our2tgt = []
for srcidx, ex in enumerate(all_gt_ex_dicts):
    map_our2tgt.append(id2tgtidx[ex['id']])
# original order -> our order
map_tgt2our = [None] * len(map_our2tgt)
for our_idx, tgt_idx in enumerate(map_our2tgt):
    map_tgt2our[tgt_idx] = our_idx
for i in map_tgt2our:
    assert i != None

print("Dumping sorted index mappings")
dump_path = '/export/share/rmeng/data/json/cnndm/our2tgt.idxmap.json'
json.dump(map_our2tgt, open(dump_path, 'w'))
dump_path = '/export/share/rmeng/data/json/cnndm/tgt2our.idxmap.json'
json.dump(map_tgt2our, open(dump_path, 'w'))


print("Dumping sorted word test.jsonl")
our_jsonl_path = "/export/share/rmeng/output/word/tokenized/cnndm/test.jsonl"
our_resorted_path = "/export/share/rmeng/output/word/tokenized/cnndm/test.sorted.jsonl"
our_lines = open(our_jsonl_path, 'r').readlines()
with open(our_resorted_path, 'w') as our_resorted:
    for tgt_idx in range(len(our_lines)):
        our_resorted.write(our_lines[map_tgt2our[tgt_idx]])

print("Dumping sorted roberta-base test.jsonl")
our_jsonl_path = "/export/share/rmeng/output/roberta-base/tokenized/cnndm/test.jsonl"
our_resorted_path = "/export/share/rmeng/output/roberta-base/tokenized/cnndm/test.sorted.jsonl"
our_lines = open(our_jsonl_path, 'r').readlines()
with open(our_resorted_path, 'w') as our_resorted:
    for tgt_idx in range(len(our_lines)):
        our_resorted.write(our_lines[map_tgt2our[tgt_idx]])
