'''
filter MAG data by fos (field of study)
'''
import argparse
import json
import os

import logging
import re
from functools import partial

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

cs_fos = set(
    ['computer science', 'algorithm', 'artificial intelligence', 'bioinformatics',
    'computational science', 'computer architecture', 'computer engineering', 'computer graphics',
    'computer hardware', 'computer network', 'computer security', 'computer vision',
    'data mining', 'data science', 'database', 'distributed computing',
    'embedded system', 'embedded system', 'electronic engineering', 'electrical engineering' 
     'human–computer interaction',
    'information retrieval', 'internet privacy', 'knowledge management', 'library science',
    'machine learning', 'multimedia', 'natural language processing',
    'operating system', 'parallel computing', 'pattern recognition', 'programming language',
    'real-time computing', 'simulation', 'software engineering', 'speech recognition',
    'telecommunications', 'theoretical computer science', 'text mining',
     'world wide web']
)

def mag2kp(mag_ex, id_field, title_field, text_field, keyword_field):
    if title_field not in mag_ex or text_field not in mag_ex:
        return None

    id_str = mag_ex[id_field]

    title = mag_ex[title_field].strip()
    abstract = mag_ex[text_field].strip()

    # split keywords to a list
    if keyword_field in mag_ex:
        keyphrases = [k.strip() for k in mag_ex[keyword_field]]
    else:
        keyphrases = []

    example = {
        "id": id_str,
        "title": title,
        "abstract": abstract,
        "keywords": keyphrases,
        "fos": mag_ex['fos'],
    }

    return example

def extract_papers(input_dir, output_dir, chunk_size, lang, must_have_kp=False):
    file_count = 0
    paper_count = 0
    domain_paper_count = 0
    assert chunk_size > 0

    mag2kp_fn = partial(mag2kp,
                        id_field = 'id',
                        title_field = 'title',
                        text_field = 'abstract',
                        keyword_field = 'keywords')

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    file_list = [fn for fn in os.listdir(input_dir) if fn.startswith('mag_papers_')]
    file_list = sorted(file_list, key=lambda x: int(re.search('mag_papers_(.*?)\.txt', x).group(1)))
    output_file_path = os.path.join(output_dir, 'train_%d.json' % (domain_paper_count // chunk_size))
    output_file = open(output_file_path, 'w')

    try:
        for txt_file in file_list:
            input_file_path = os.path.join(input_dir, txt_file)
            print(input_file_path)
            file_count += 1

            with open(input_file_path, 'r') as input_file:
                for line in input_file:
                    paper_count+=1
                    if paper_count % 10000==0:
                        logging.info('The {:} th File:{:}, total progress: {:}/{:} papers in CS fos'.format(file_count, input_file_path, domain_paper_count, paper_count))
                    mag_ex = json.loads(line)
                    if 'fos' not in mag_ex or mag_ex.get('lang') != lang:
                        continue

                    if must_have_kp and 'keywords' not in mag_ex:
                        continue

                    is_cs = any([True if f.lower().strip() in cs_fos else False for f in mag_ex['fos']])
                    if is_cs:
                        kp_ex = mag2kp_fn(mag_ex)
                        if kp_ex is None:
                            continue
                        output_file.write(json.dumps(kp_ex)+'\n')
                        domain_paper_count += 1

                        if domain_paper_count % chunk_size == 0:
                            output_file.close()
                            output_file_path = os.path.join(output_dir, 'train_%d.json' % (domain_paper_count // chunk_size))
                            output_file = open(output_file_path, 'w')
    finally:
        output_file.close()
    logging.info('Process finished \n\t'
                 'Find {:}/{:} CS papers in {:} MAG files: {}\n\t'
                 'Dumped to {}'.format(domain_paper_count, paper_count, file_count, input_dir, output_dir
    ))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-mag_input_dir', required=True)
    parser.add_argument('-mag_output_dir', required=True)
    parser.add_argument('-must_have_kp', action='store_true')
    parser.add_argument('-chunk_size', default=1000000, type=int)
    parser.add_argument('-lang', required=True)

    opt = parser.parse_args()

    extract_papers(opt.mag_input_dir, opt.mag_output_dir, opt.chunk_size, opt.lang, must_have_kp=opt.must_have_kp)

    print('[Info] Dumping the processed data to new text file', opt.mag_output_dir)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
