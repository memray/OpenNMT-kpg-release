'''
The data used by previous studies is sorted following the order as in wayback_test_urls.txt, where only urls are available.
Our data is sorted in a different order but the original .story filename and doc id are preserved.
Here we read urls in wayback_test_urls.txt, encode them with hashhex, and sort our data to be consistent with others.
So this script is not useful any more.
Note that CNN and DM are processed separately and merged manually. And it seems .question is hashed in a different way.
(deprecated) ```Instead, our data is generated directly from questions/test/, which is not deliberately ordered and only doc_id is available. Here we read each .question in dir `questions/test/` and extract the mapping between `id` and `url`, then we can put our data in the same order.```
'''
import hashlib
import json
import os
from collections import defaultdict

import tqdm

# dataset = 'cnn'
dataset = 'dailymail'
base_dir = '/export/share/rmeng/data/raw/cnndm/original/%s/' % dataset
test_url_path = os.path.join(base_dir, 'wayback_test_urls.txt')
test_questions_dir = os.path.join(base_dir, 'questions', 'test')
test_story_dir = os.path.join(base_dir, 'stories')

output_path = '/export/share/rmeng/data/json/cnndm/'


def hashhex(s):
    '''Returns a heximal formated SHA1 hash of the input string.'''
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]

if __name__ == '__main__':
    print('Processing dataset %s' % dataset)
    # load all urls from wayback_test_urls.txt, order of which we should map our data to
    url_list = [u.strip() for u in open(test_url_path, 'r').readlines()]
    print('#(url)=%d' % len(url_list))

    # obtain the hash code of url
    hash_list = get_url_hashes(url_list)
    for hashid in hash_list:
        story_path = os.path.join(test_story_dir, hashid + '.story')
        if os.path.exists(story_path):
            # print('Yay')
            pass
        else:
            print('Nay')
    print('Done')

    """
    # load .question files to build the mapping between unique docids (a hash code) and urls
    # one url can have multiple hashcodes (don't know why)
    url2id_map = defaultdict(list)
    for q_file in tqdm.tqdm(os.listdir(test_questions_dir), desc='Loading question files'):
        if not q_file.endswith('.question'):
            continue
        docid = '-'.join([dataset, 'test', q_file.split('.')[0]])
        url = open(os.path.join(test_questions_dir, q_file), 'r').readline().strip()
        url2id_map[url].append(docid)

    print('#(map)=%d' % len(url2id_map))

    # we write the mapping to file ($output_path/cnn_test_url2id.jsonl) following the order of wayback_test_urls.txt
    # note that dailymail has 53,182 questions, more than len(wayback_test_urls.txt), so needs filtering first
    mapping_dicts = []
    for url in url_list:
        if url not in url2id_map:
            print(url)
        mapping_dicts.append({'url': url, 'id': url2id_map[url]})

    output_jsonl_path = os.path.join(output_path, '%s_test_urlhash_mapping.jsonl' % dataset)
    with open(output_jsonl_path, 'w') as output_jsonl:
        for md in mapping_dicts:
            output_jsonl.write(json.dumps(md) + '\n')

    print('Done!')
    """