# -*- coding: utf-8 -*-
"""
Remove noisy items (abstract contains "Full textFull text is available as a scanned copy of the original print version.") (around 132561 out of 3114539=2981978) and remove duplicates by title
"""
import codecs
import json
import os
import re
import string

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def example_iterator_from_json(path, dataset_name, id_field, title_field, text_field, keyword_field, trg_delimiter=';', is_train=False):
    '''
    Load id/title/abstract/keyword, don't do any preprocessing
    ID is required to match the original data
    '''
    global valid_num
    print("Loading %s" % os.path.abspath(path))

    with codecs.open(path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(corpus_file):
            # if(idx == 2000):
            #     break
            # print(line)

            _json = json.loads(line)

            if id_field is None or id_field not in _json:
                id_str = '%s_%d' % (dataset_name, idx)
            else:
                id_str = _json[id_field]

            if is_train and title_field not in _json or keyword_field not in _json or text_field not in _json:
                # print("Data is missing:\n%s" % (_json))
                continue

            title_str = _json[title_field].strip(string.punctuation)
            abstract_str = _json[text_field].strip(string.punctuation)

            # split keywords to a list
            if trg_delimiter:
                keyphrase_strs = [k.strip(string.punctuation) for k in re.split(trg_delimiter, _json[keyword_field])
                                  if len(k.strip(string.punctuation)) > 0]
            else:
                keyphrase_strs = [k.strip(string.punctuation) for k in _json[keyword_field]
                                  if len(k.strip(string.punctuation)) > 0]

            if is_train and abstract_str.startswith('Full textFull text'):
                continue

            if is_train and len(title_str) == 0 or len(abstract_str) == 0 or len(keyphrase_strs) == 0:
                continue

            example = {
                "id": id_str,
                "title": title_str,
                "abstract": abstract_str,
                "keywords": keyphrase_strs,
            }

            valid_num += 1
            yield example

if __name__ == '__main__':
    mag_path = "source_data/mag_output/mag_nodup.json"
    mag_output_path = "source_data/mag_output/mag_nodup_plus.json"
    kp20k_train_path = "source_data/kp20k/kp20k_training.json"

    train_dataset_name = 'mag'
    test_dataset_names = ['kp20k_train']
    id_field = 'id'
    title_field = 'title'
    text_field = 'abstract'
    keyword_field = 'keywords'
    trg_delimiter = None

    mag_examples_iter = list(example_iterator_from_json(path=mag_path,
                                                     dataset_name="mag",
                                                     id_field=id_field,
                                                     title_field=title_field,
                                                     text_field=text_field,
                                                     keyword_field=keyword_field,
                                                     trg_delimiter=trg_delimiter))
    print("Loaded %d examples from MAG" % len(mag_examples_iter))

    id_field = None
    keyword_field = 'keywords'
    trg_delimiter = ';'
    kp20k_train_examples = list(example_iterator_from_json(path=kp20k_train_path,
                                                    dataset_name="kp20k_train",
                                                    id_field=id_field,
                                                    title_field=title_field,
                                                    text_field=text_field,
                                                    keyword_field=keyword_field,
                                                    trg_delimiter=trg_delimiter))

    print("Loaded %d examples from KP20k train" % len(kp20k_train_examples))

    title_pool = set()
    for ex in kp20k_train_examples:
        title_pool.add(ex["title"].lower().strip())

    non_dup_count = 0
    with open(mag_output_path, 'w') as mag_output:
        for ex_id, ex in enumerate(mag_examples_iter):
            title = ex["title"].lower().strip()
            if title not in title_pool:
                non_dup_count += 1
                title_pool.add(title)
                mag_output.write(json.dumps(ex) + '\n')
                if ex_id % 1000 == 0:
                    print("non-dup/processed/all = %d/%d/%d" % (non_dup_count, ex_id, len(mag_examples_iter)))

