# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import codecs
import json
import os

from nltk.tag import StanfordPOSTagger
from nltk.internals import find_jars_within_path
import re
import six

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.abstract       = ''
        self.fulltext       = ''
        self.keyword    = []

    def __str__(self):
        return '%s\n\t%s\n\t%s\n\t%s' % (self.name, self.title, self.abstract, str(self.keyword))

    def to_dict(self):
        d = {}
        d['name'] = self.name
        d['title'] = re.sub('[\r\n]', ' ', self.title).strip()
        d['abstract'] = re.sub('[\r\n]', ' ', self.abstract).strip()
        d['fulltext'] = self.fulltext
        d['keyword'] = ';'.join(self.keyword)
        return d


class Dataset(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name    = self.__class__.__name__.lower()
        self.datadir = os.path.join(basedir, self.name.lower())
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'
        self.train_test_splitted = False
        self.title_abstract_body_separated = False

        self.doc_list = []


    def load_dataset(self):
        self.load_text(self.textdir)
        self.load_keyphrase(self.keyphrasedir)


    def load_dataset_as_dicts(self):
        return self._convert_docs_to_dicts(self.doc_list)


    def _convert_docs_to_dicts(self, docs):
        '''
        :return: a list of dict
        '''
        dict_list = []
        for d in docs:
            dict_list.append(d.to_dict())

        return dict_list


    def load_train_test_dataset(self):
        train_data = []
        test_data = []
        if self.train_test_splitted:
            '''
            if the split of train/test is given, return as it is
            '''
            for doc in self.doc_list:
                if doc.name.startswith('train'):
                    train_data.append(doc)
                elif doc.name.startswith('test'):
                    test_data.append(doc)
                else:
                    raise Exception('File must start with either train or test if train_test_splitted is on for class %s' % self.__class__)
        else:
            '''
            if split is not given, take the first 20% for test, rest 80% for training
            '''
            # ensure files are sorted in an alphabetical order
            doc_list = sorted(self.doc_list, key=lambda d:d.name)
            test_data = doc_list[: int(len(doc_list) * 0.2)]
            train_data = doc_list[int(len(doc_list) * 0.2): ]

        train_data_dicts = self._convert_docs_to_dicts(train_data)
        test_data_dicts = self._convert_docs_to_dicts(test_data)

        return train_data_dicts, test_data_dicts


    def dump_train_test_to_json(self):
        train_data_dicts, test_data_dicts = self.load_train_test_dataset()
        train_json_path = os.path.join(self.datadir, self.name.lower() + '_train.json')
        with open(train_json_path, 'w') as train_json:
            for d in train_data_dicts:
                train_json.write(json.dumps(d) + '\n')

        test_json_path = os.path.join(self.datadir, self.name.lower() + '_test.json')
        with open(test_json_path, 'w') as test_json:
            for d in test_data_dicts:
                test_json.write(json.dumps(d) + '\n')


    def load_text(self, textdir):
        # ensure files are loaded in an alphabetical order
        file_names = os.listdir(textdir)
        file_names = sorted(file_names)

        for fid, filename in enumerate(file_names):
            # with codecs.open(textdir+filename, "r", encoding='utf-8', errors='ignore') as textfile:
            with open(textdir+filename) as textfile:
                try:
                    lines = textfile.readlines()
                    lines = [line.strip() for line in lines]

                    if self.title_abstract_body_separated:
                        '''
                        title/abstract/fulltext are separated by --T/--A/--B
                        '''
                        T_index = None
                        for line_id, line in enumerate(lines):
                            if line.strip() == '--T':
                                T_index = line_id
                                break

                        A_index = None
                        for line_id, line in enumerate(lines):
                            if line.strip() == '--A':
                                A_index = line_id
                                break

                        B_index = None
                        for line_id, line in enumerate(lines):
                            if line.strip() == '--B':
                                B_index = line_id
                                break

                        # lines between T and A are title
                        title = ' '.join(lines[T_index + 1: A_index])
                        # lines between A and B are abstract
                        abstract = ' '.join(lines[A_index + 1: B_index])
                        # lines after B are fulltext
                        fulltext = '\n'.join(lines[B_index + 1:])

                        if T_index is None or A_index is None or B_index is None:
                            print('Wrong format detected : %s' % (filename))
                            print('Name: ' + textdir + filename)
                            print('Title: ' + title.strip())
                            if not T_index:
                                print('line 0 should be --T: ' + ''.join(lines[0]).strip())
                            if not A_index:
                                print('line 2 should be --A: ' + ''.join(lines[2]).strip())
                            if not B_index:
                                print('line 4 should be --B: ' + ''.join(lines[4]).strip())
                            print()
                        else:
                            pass
                            # print('No Problem: %s' % filename)

                    else:
                        '''
                        otherwise, 1st line is title, and rest lines are abstract
                        '''

                        # 1st line is title
                        title = lines[0]
                        # rest lines are abstract
                        abstract = (' '.join([''.join(line).strip() for line in lines[1:]]))
                        # no fulltext is given, ignore it
                        fulltext = ''

                    doc = Document()
                    doc.name = filename[:filename.find('.txt')]
                    doc.title = title
                    doc.abstract = abstract
                    doc.fulltext = fulltext
                    self.doc_list.append(doc)

                except UnicodeDecodeError as e:
                    print('UnicodeDecodeError detected! %s' % (textdir+filename))
                    print(e)


    def load_keyphrase(self, keyphrasedir):
        for did,doc in enumerate(self.doc_list):
            phrase_set = set()

            if os.path.exists(self.keyphrasedir + doc.name + '.keyphrases'):
                with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            if os.path.exists(self.keyphrasedir + doc.name + '.keywords'):
                with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            doc.keyword = list(phrase_set)


class INSPEC(Dataset):
    def __init__(self, **kwargs):
        super(INSPEC, self).__init__(**kwargs)
        self.train_test_splitted = True


class NUS(Dataset):
    def __init__(self, **kwargs):
        super(NUS, self).__init__(**kwargs)
        self.title_abstract_body_separated = True


class SemEval(Dataset):
    def __init__(self, **kwargs):
        super(SemEval, self).__init__(**kwargs)
        self.train_test_splitted = True
        self.title_abstract_body_separated = True


class KRAPIVIN(Dataset):
    def __init__(self, **kwargs):
        super(KRAPIVIN, self).__init__(**kwargs)
        self.title_abstract_body_separated = True


class DUC(Dataset):
    def __init__(self, **kwargs):
        super(DUC, self).__init__(**kwargs)


# aliases
inspec = INSPEC
nus = NUS
semeval = SemEval
krapivin = KRAPIVIN
duc = DUC


def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


def initialize_test_data_loader(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    test_data = get_from_module(identifier.lower(), globals(), 'data_loader', instantiate=True,
                           kwargs=kwargs)
    return test_data


PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'
SEP_WORD = '<sep>'


def copyseq_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))

    # replace the digit terms with <digit>
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens


def load_pos_tagger(stanford_base_dir):
    # path = os.path.dirname(__file__)
    # path = os.path.join(file_dir[: file_dir.rfind('pykp') + 4], 'stanford-postagger')
    # print(path)
    # jar = '/Users/memray/Project/stanford/stanford-postagger/stanford-postagger.jar'
    jar = stanford_base_dir + '/stanford-postagger.jar'
    model = stanford_base_dir + '/models/english-bidirectional-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model_filename=model, path_to_jar=jar)

    stanford_base_dir = jar.rpartition('/')[0]
    stanford_jars = find_jars_within_path(stanford_base_dir)
    pos_tagger._stanford_jar = ':'.join(stanford_jars)

    return pos_tagger


extra_dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'duc']
def export_extra_dataset_to_json():
    for dataset_name in extra_dataset_names:
        print('-' * 50)
        print('Loading %s' % dataset_name)

        dataset_loader = initialize_test_data_loader(dataset_name)
        dataset_loader.load_dataset()
        dataset_dict = dataset_loader.load_dataset_as_dicts()
        train_data_dicts, test_data_dicts = dataset_loader.load_train_test_dataset()
        dataset_loader.dump_train_test_to_json()

        print('#(doc) = %d' % (len(dataset_dict)))
        print('#(keyphrase) = %.3f' % (sum([len(d.keyword) for d in dataset_loader.doc_list]) / len(dataset_dict)))
        print('#(train) = %d, #(test)=%d' % (len(train_data_dicts), len(test_data_dicts)))

        print('\nlen(title) = %.3f' % (sum([len(d.title.split()) for d in dataset_loader.doc_list]) / len(dataset_dict)))
        print('len(abstract) = %.3f' % (sum([len(d.abstract.split()) for d in dataset_loader.doc_list]) / len(dataset_dict)))
        print('len(fulltext) = %.3f' % (sum([len(d.fulltext.split()) for d in dataset_loader.doc_list]) / len(dataset_dict)))

        # print(dataset_loader.doc_list[10])
        # print(dataset_loader.doc_list[20])



def add_testset_postag(test_dataset_names, dataset_base_dir, stanford_base_dir):
    pos_tagger = load_pos_tagger(stanford_base_dir)

    for dataset_name in test_dataset_names:
        abstract_key = 'abstract'
        if dataset_name =='stackexchange':
            abstract_key = 'question'

        print('-' * 50)
        print('Loading %s' % dataset_name)
        json_path = os.path.join(dataset_base_dir, dataset_name, dataset_name+'_test.json')

        dataset_dict_list = []
        # load from json file
        with open(json_path, 'r') as json_file:
            for line in json_file:
                dataset_dict_list.append(json.loads(line))

        json_path = os.path.join(dataset_base_dir, dataset_name, dataset_name + '_test_postag.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                lines = [l for l in json_file if len(l) > 0]
                if len(lines) != len(dataset_dict_list):
                    print("Number of previous results doesn't match original dataset")
                else:
                    print('%s seems already processed, skip!' % dataset_name)
                    continue
        else:
            print('Processing and dumping to %s' % dataset_name)
        # dump to another json
        with open(json_path, 'w') as json_file:
            # postag title/abstract and insert into data example
            for e_id, example_dict in enumerate(dataset_dict_list):
                print('=' * 50)
                print(e_id)
                print(example_dict['title'])
                print('len(title)=%d' % len(example_dict['title']))
                print('len(abstract)=%d' % len(example_dict[abstract_key]))

                if len(example_dict[abstract_key]) > 1000:
                    print('truncate to 1000 words')
                    example_dict[abstract_key] = example_dict[abstract_key][:1000]

                if e_id % 10 == 0:
                    print('Processing %d/%d' % (e_id, len(dataset_dict_list)))

                title_postag_tokens = pos_tagger.tag(copyseq_tokenize(example_dict['title']))
                print('#(title token)=%d : %s' % (len(title_postag_tokens), str(title_postag_tokens)))
                abstract_postag_tokens = pos_tagger.tag(copyseq_tokenize(example_dict[abstract_key]))
                print('#(abstract token)=%d : %s' % (len(abstract_postag_tokens), str(abstract_postag_tokens)))
                example_dict['title_postag'] = ' '.join([str(t[0])+'_'+str(t[1]) for t in title_postag_tokens])
                example_dict['abstract_postag'] = ' '.join([str(t[0])+'_'+str(t[1]) for t in abstract_postag_tokens])

                # for example_dict in postag_dataset_dict_list:
                json_file.write(json.dumps(example_dict) + '\n')


def export_to_UTD_format(dataset_name, dataset_base_dir, dump_path):
    # export test data only
    plain_text_dir = os.path.join(dump_path, 'plain_text')
    postag_text_dir = os.path.join(dump_path, 'text')
    keyphrase_dir = os.path.join(dump_path, 'keyphrase')
    list_file_path = os.path.join(dump_path, '%s_list.txt' % dataset_name)
    dirs = [plain_text_dir, postag_text_dir, keyphrase_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    title_key = 'title'
    abstract_key = 'abstract'
    keyword_key = 'abstract'
    if dataset_name == 'stackexchange':
        abstract_key = 'question'
        keyword_key = 'tags'

    text_filename_list = []
    data_json_path = os.path.join(dataset_base_dir, dataset_name, dataset_name + '_test_postag.json')

    with open(data_json_path, 'r') as data_json_file:
        for doc_id, line in enumerate(data_json_file):
            example_dict = json.loads(line)
            text_filename_list.append('%s.txt' % doc_id)
            with open(os.path.join(plain_text_dir, '%s.txt' % doc_id), 'w') as plain_text:
                plain_text.write(example_dict[title_key] + '.\n' + example_dict[abstract_key])
            with open(os.path.join(postag_text_dir, '%s.txt' % doc_id), 'w') as postag_text:
                postag_text.write(example_dict['title_postag'] + ' ._. ' + example_dict['abstract_postag'])
            with open(os.path.join(keyphrase_dir, '%s.txt' % doc_id), 'w') as keyphrase_file:
                for target in example_dict[keyword_key].split(';'):
                    keyphrase_file.write(target+'\n')

    with open(list_file_path, 'w') as list_file:
        for text_filename in text_filename_list:
            list_file.write(text_filename+'\n')

    print('Export to UTD format done!')


def export_to_maui_format(dataset_name, dataset_base_dir, dump_path, mode=None):
    assert mode in ['kea', 'maui']
    title_key = 'title'
    abstract_key = 'abstract'
    keyword_key = 'abstract'
    if dataset_name == 'stackexchange':
        abstract_key = 'question'
        keyword_key = 'tags'

    # export both training and testing
    for data_type in ['train', 'test']:
        dump_dir_path = os.path.join(dump_path, data_type)
        if not os.path.exists(dump_dir_path):
            os.makedirs(dump_dir_path)

        # only train KEA and Maui with 50k data, tried 100k and they OOM all the time
        train_limit = 50000
        # only 'kp20k' and 'stackexchange' have training data
        if data_type=='train':
            if dataset_name != 'kp20k' and dataset_name != 'stackexchange':
                continue
        data_json_path = os.path.join(dataset_base_dir, dataset_name, dataset_name + '_%s.json' % data_type)

        with open(data_json_path, 'r') as data_json_file:
            for doc_id, line in enumerate(data_json_file):
                if data_type == 'train' and doc_id >= train_limit:
                    break
                example_dict = json.loads(line)
                with open(os.path.join(dump_dir_path, '%s.txt' % doc_id), 'w') as text_file:
                    text_file.write(example_dict[title_key] + '.\n' + example_dict[abstract_key])
                with open(os.path.join(dump_dir_path, '%s.key' % doc_id), 'w') as key_file:
                    for target in example_dict[keyword_key].split(';'):
                        if mode == 'maui':
                            key_file.write(target + '\t1\n')
                        elif mode == 'kea':
                            key_file.write(target + '\n')
                        else:
                            raise Exception('Wrong mode')

    print('Export to %s format done!' % mode)


if __name__ == '__main__':
    dataset_names = ['stackexchange', 'inspec', 'nus', 'semeval', 'krapivin', 'duc', 'kp20k']
    dataset_json_dir = '/Users/memray/project/kp/OpenNMT-kpg/data/keyphrase/json/'
    # export_extra_dataset_to_json()
    add_testset_postag(dataset_names,
                       dataset_base_dir=dataset_json_dir,
                       stanford_base_dir='/Users/memray/project/stanford/stanford-postagger/')

    for dataset_name in dataset_names:
        print('-=*=-' * 10)
        print("Exporting %s" % dataset_name)
        # dump_path = '/Users/memray/project/kp/OpenNMT-kpg/data/keyphrase/baseline/utd/%s/' % dataset_name
        # export_to_UTD_format(dataset_name, dataset_json_dir, dump_path)

        dump_path = '/Users/memray/project/kp/OpenNMT-kpg/data/keyphrase/baseline/maui/%s/' % dataset_name
        export_to_maui_format(dataset_name, dataset_json_dir, dump_path, mode='maui')

        dump_path = '/Users/memray/project/kp/OpenNMT-kpg/data/keyphrase/baseline/kea/%s/' % dataset_name
        export_to_maui_format(dataset_name, dataset_json_dir, dump_path, mode='kea')