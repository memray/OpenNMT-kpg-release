# -*- coding: utf-8 -*-
import json
import re
from functools import partial

import six
import torch
import numpy as np
from torchtext.data import Field, RawField
from tqdm import tqdm

from onmt.keyphrase.utils import if_present_duplicate_phrases, SEP_token, DIGIT_token
from onmt.inputters.datareader_base import DataReaderBase

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example

from onmt.inputters.dataset_base import _join_dicts, _dynamic_dict

np.random.seed(2333)
extra_special_tokens = [SEP_token, DIGIT_token]

KP_DATASET_FIELDS = {'scipaper': ('title', 'abstract', 'keywords', None),
                     'qa': ('title', 'question', 'tags', 'categories'),
                     'webpage': ('url', 'text', 'KeyPhrases', None),
                     'news': ('title', 'abstract', 'keyword', 'categories')}

KP_CONCAT_TYPES = ['one2one', 'random',
                   'pres_abs', 'abs_pres',
                   'nosort', 'nosort_reverse',
                   'alphab', 'alphab_reverse',
                   'length', 'length_reverse']
# previous version (used in empirical study)
# KP_CONCAT_TYPES = ['random',
#             'no_sort', 'no_sort_reverse',
#             'alphabetical', 'alphabetical_reverse',
#             'length', 'length_reverse',
#             'verbatim_append', 'verbatim_prepend']

def infer_dataset_type(filepath):
    dataset_type = None
    if 'stack' in filepath:
        dataset_type = 'qa'
    elif 'openkp' in filepath:
        dataset_type = 'webpage'
    elif 'kptimes' in filepath or 'jptimes' in filepath:
        dataset_type = 'news'
    elif 'kp20k' in filepath or 'magkp' in filepath:
        dataset_type = 'scipaper'
    elif 'duc' in filepath or 'inspec' in filepath or 'krapivin' in filepath \
            or 'nus' in filepath or 'semeval' in filepath:
        # duc is an outlier
        dataset_type = 'scipaper'

    assert dataset_type is not None, 'Fail to detect the data type of the given input file.' \
                                     'Accecpted values:' + KP_DATASET_FIELDS.keys()

    print('Automatically detect the input data type as [%s], path: %s' % (dataset_type.upper(), filepath))

    return dataset_type


def parse_src_fn(ex_dict, title_field, text_field):
    concat_str = ex_dict[title_field] + ' . ' + ex_dict[text_field]
    return concat_str


def kpdict_parse_fn(example, tokenizer, tgt_concat_type, dataset_type='scipaper', max_target_phrases=-1, lowercase=False):
    assert dataset_type in KP_DATASET_FIELDS
    title_field, text_field, keyword_field, category_field = KP_DATASET_FIELDS[dataset_type]

    src_str = parse_src_fn(example, title_field, text_field)
    if isinstance(example[keyword_field], str):
        tgt_kps = example[keyword_field].split(';')
    else:
        tgt_kps = example[keyword_field]
    if tgt_concat_type == 'one2one':
        # sample one tgt from multiple tgts and use it as the only tgt
        rand_idx = np.random.randint(len(tgt_kps))
        tgt_str = tgt_kps[rand_idx]
    elif tgt_concat_type in KP_CONCAT_TYPES:
        # generate one2seq training data points
        order = obtain_sorted_indices(src_str.lower().split(),
                                      [kp.lower().split() for kp in tgt_kps],
                                      sort_by=tgt_concat_type)
        if max_target_phrases > 0 and len(order) > max_target_phrases:
            order = order[: max_target_phrases]
        tgt = [tgt_kps[idx] for idx in order]
        tgt_str = tokenizer.sep_token.join(tgt)
    else:
        raise NotImplementedError('Unsupported target concatenation type ' + tgt_concat_type)

    if lowercase:
        return src_str.lower(), tgt_str.lower()
    return src_str, tgt_str


class KeyphraseDataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, List[Tuple[str, Field]]]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None, tgt_concat_type=None, data_format=None, max_target_phrases=-1):
        # if not using pre-trained tokenizer, this is set at line 594 in inputter.py and line 303 in translator.py
        self.tgt_type = tgt_concat_type
        # concatenate multiple tgt sequences with <sep> or keep them separate as a list of seqs (2D tensor)
        # self.concat_tgt = False
        self.data_format = data_format
        self.sort_key = sort_key
        self.max_target_phrases = max_target_phrases

        # will be specified before training, one of [one2one, original, random, verbatim]

        # build src_map/alignment no matter field is available
        can_copy = True

        self.dataset_type = infer_dataset_type(dirs[0])
        dirs = [None, None]
        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is temp vocab/alignment, used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in tqdm(starmap(_join_dicts, zip(*read_iters)), desc='Loading and parsing data'):
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']

                if hasattr(src_field, 'pretrained_tokenizer'):
                    tokenizer = src_field.pretrained_tokenizer

                if data_format == 'jsonl':
                    title_field, text_field, keyword_field, _ = KP_DATASET_FIELDS[self.dataset_type]
                    src_str, tgt_str = kpdict_parse_fn(ex_dict['src'], tokenizer, tgt_concat_type,
                                                       dataset_type=self.dataset_type, max_target_phrases=-1,
                                                       lowercase=False)
                    ex_dict['src'] = src_str
                    ex_dict['tgt'] = tgt_str

                    # src_str = parse_src_fn(ex_dict['src'], title_field, text_field)
                    # if isinstance(ex_dict['src'][keyword_field], str):
                    #     tgt_kps = ex_dict['src'][keyword_field].split(';')
                    # else:
                    #     tgt_kps = ex_dict['src'][keyword_field]
                    # ex_dict['src'] = src_str
                    # ex_dict['tgt'] = ';'.join(tgt_kps)

                # this assumes src_field and tgt_field are both text
                src_base_field = src_field.base_field if hasattr(src_field, 'base_field') else src_field
                tgt_base_field = tgt_field.base_field if hasattr(tgt_field, 'base_field') else tgt_field
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_base_field, tgt_base_field)
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(KeyphraseDataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    def load_config(self, opt):
        self.tgt_type = opt.tgt_type


class KeyphraseDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read keyphrase data from disk. Current supported data format is JSON only.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
            src: title+abstract
            tgt: a string of a keyword, or a string of concatenated keywords (delimited by <sep>)
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with KeyphraseDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, line in enumerate(sequences):
            try:
                # default input is a json line
                line = line.decode("utf-8")
                json_dict = json.loads(line)
                try:
                    # Note tgt could be a list of strings
                    seq = json_dict[side]
                    # torchtext field only takes numeric features
                    id = json_dict['id']
                except Exception:
                    # temporary workaround for raw json data
                    seq = json_dict
                    id = i
            except Exception:
                # temporary workaround for plain text input
                seq = line
                id = i

            try:
                if id.rfind('_') != -1:
                    id = id[id.rfind('_') + 1:]
                id = int(id)
            except Exception:
                # if not convertible, use indices as id
                id = i

            yield {side: seq, "indices": i, 'id': id}


def obtain_sorted_indices(src, tgt_seqs, sort_by):
    """
    :param src: used for verbatim and alphabetical
    :param tgt_seqs:
    :param sort_by:
    :param absent_pos: must be one of [prepend, append and ignore], ignore means simply drop absent kps
    :return:
    """
    # TODO: change to spacy tokenization
    print('TODO: change to spacy tokenization!!!')
    num_tgt = len(tgt_seqs)

    if sort_by == 'random':
        sorted_id = np.random.permutation(num_tgt)
    elif sort_by.startswith('nosort'):
        sorted_id = list(range(len(tgt_seqs)))
    elif sort_by.startswith('alphab'):
        sorted_tgts = sorted(enumerate(tgt_seqs), key=lambda x: '_'.join(x[1]))
        sorted_id = [t[0] for t in sorted_tgts]
    elif sort_by.startswith('length'):
        sorted_tgts = sorted(enumerate(tgt_seqs), key=lambda x: len(x[1]))
        sorted_id = [t[0] for t in sorted_tgts]
    elif sort_by == 'pres_abs' or sort_by == 'abs_pres':
        # obtain present flags as well their positions, lowercase should be done beforehand
        present_tgt_flags, present_indices, _ = if_present_duplicate_phrases(src, tgt_seqs)
        # separate present/absent phrases
        present_tgt_idx = np.arange(num_tgt)[present_tgt_flags]
        absent_tgt_idx  = [t_id for t_id, present in zip(range(num_tgt), present_tgt_flags) if ~present]
        absent_tgt_idx  = np.random.permutation(absent_tgt_idx)
        # sort present phrases by their positions
        present_indices = present_indices[present_tgt_flags]
        present_tgt_idx = sorted(zip(present_tgt_idx, present_indices), key=lambda x: x[1])
        present_tgt_idx = [t[0] for t in present_tgt_idx]

        if sort_by == 'pres_abs':
            sorted_id = np.concatenate((present_tgt_idx, absent_tgt_idx), axis=None)
        elif sort_by == 'abs_pres':
            sorted_id = np.concatenate((absent_tgt_idx, present_tgt_idx), axis=None)
        else:
            raise NotImplementedError('Unsupported sort_by value: ' + sort_by)
            sorted_id = present_tgt_idx
    else:
        raise NotImplementedError('Unsupported sort_by value: ' + sort_by)

    if sort_by.endswith('reverse'):
        sorted_id = sorted_id[::-1]

    return np.asarray(sorted_id, dtype=int)


def process_multiple_tgts(big_batch, tgt_type):
    """
    This function is only used using original OpenNMT pipeline (data stored in .pt file).
    If data is loaded from raw JSON with pre-trained tokenizer, it is processed on-the-fly.
    :param big_batch: a list of examples
            src: [1, src_len]
            tgt: [num_kp, 1, kp_len]
    :param tgt_type:
            specify format of target and concatenate kps in tgt accordingly
            if one2one: randomly pick up one phrase, tgt will be [1, kp_len]
            if one2seq: tgt will be [1, concat_kps_len]
    :return:
    """
    new_batch = []
    for ex in big_batch:
        # a workaround: truncate to maximum 8 phrases (some noisy data points have many phrases)
        if hasattr(ex, "tgt") and len(ex.tgt) > 8:
            random_choise = np.random.choice(len(ex.tgt), 8)
            if hasattr(ex, "tgt"):
                ex.tgt = [ex.tgt[idx] for idx in random_choise]
            if hasattr(ex, "alignment"):
                ex.alignment = [ex.alignment[idx] for idx in random_choise]
        # tgt = ex.tgt if hasattr(ex, "tgt") else None
        # alignment = ex.alignment if hasattr(ex, "alignment") else None

        # sep_indices, indicating the position of <SEP> and <EOS> after concatenating, only used in one2seq training
        sep_indices = None
        if tgt_type == 'one2one':
            # sample one tgt from multiple tgts and use it as the only tgt
            rand_idx = np.random.randint(len(ex.tgt))
            tgt = ex.tgt[rand_idx]
            alignment = ex.alignment[rand_idx] if hasattr(ex, "alignment") else None

            ex.tgt = tgt
            ex.alignment = alignment
        elif tgt_type in KP_CONCAT_TYPES:
            # generate one2seq training data points
            order = obtain_sorted_indices(ex.src, ex.tgt, sort_by=tgt_type)
            tgt = [ex.tgt[idx] for idx in order]
            tgt = [t[0]+[SEP_token] for t in tgt[:-1]] + tgt[-1]
            tgt = [np.concatenate(tgt, axis=None).tolist()]

            # position of <SEP> and <EOS>
            sep_indices = [[wid for wid, w in enumerate(t) if w==SEP_token] + [len(t)] for t in tgt]
            sep_indices = torch.torch.from_numpy(np.concatenate(sep_indices, axis=None))

            # print("len_tgt=%d" % len(tgt[0]))
            # print("sep_indices=%s" % str(sep_indices.tolist()))

            if hasattr(ex, "alignment"):
                alignment = [ex.alignment[idx] for idx in order]
                # remove the heading and trailing 0 for <s> and </s> in each subsequence
                alignment = [a.numpy().tolist()[1:-1] for a in alignment]
                # add pads 0 for <sep> between subsequences, <s> and </s> for whole final sequence
                alignment = [[0]] + [t+[0] for t in alignment[:-1]] + [alignment[-1]] + [[0]]
                # concatenate alignments to one Tensor, length should be len(tgt)+2
                alignment = torch.torch.from_numpy(np.concatenate(alignment, axis=None))
            else:
                alignment = None

            ex.tgt = tgt
            ex.alignment = alignment
            '''
            elif tgt_type == 'no_sort':
                # return tgts in original order
                tgt = [t[0]+[SEP_token] for t in ex.tgt[:-1]] + ex.tgt[-1]
                tgt = [np.concatenate(tgt, axis=None).tolist()]
                if hasattr(ex, "alignment"):
                    # remove the heading and trailing 0 for <s> and </s>
                    alignment = [a.numpy().tolist()[1:-1] for a in ex.alignment]
                    # add 0s for <sep>, <s> and </s>
                    alignment = [0] + [t+[0] for t in alignment[:-1]] + [alignment[-1]] + [0]
                    # concatenate alignments to one Tensor
                    alignment = torch.torch.from_numpy(np.concatenate(alignment, axis=None))
                else:
                    alignment = None
            '''
        # no processing for 'multiple' (test phrase)
        elif tgt_type == 'multiple':
            pass
        else:
            raise NotImplementedError

        setattr(ex, 'sep_indices', sep_indices)

        if hasattr(ex, "alignment"):
            if isinstance(alignment, list):
                # for test phase (tgt_type='multiple'), with unprocessed multiple targets
                assert len(tgt) == len(alignment)
                assert all([len(t[0])+2==a.size()[0] for t,a in zip(tgt, alignment)])
            else:
                # for other training cases, with one target sequence
                assert len(tgt[0]) + 2 == alignment.size()[0]

        new_batch.append(ex)

    return new_batch


def kp_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])

# deprecated
def max_tok_len(new, count, sofar):
    """
    Specialized for keyphrase generation task
    Note that the form of tgt has to be determined beforehand, i.e. shuffle/order/pad should have been done
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def copyseq_tokenize(text):
    '''
    moved to onmt.keyphrase.utils.meng17_tokenize()
    '''
    pass

# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None, lower=False):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """
    if lower:
        string = string.lower()

    # @memray 20190308 to make tokenized results same between src/tgt, changed here back to a simple splitter (whitespace)
    # move complicated tokenization into pre-preprocess in kp_data_converter.py
    tokens = string.split(tok_delim)
    # tokens = copyseq_tokenize(string)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class KeyphraseField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field):
        super(KeyphraseField, self).__init__()
        self.fields = [(base_name, base_field)]
        self.type = None

    @property
    def base_field(self):
        return self.fields[0][1]

    def pad_seqs(self, batch, max_seq_num, max_seq_len, pad_token):
        """
        batch is a list of seqs (each seq is a list of strings)
        pad empty seqs to each example in batch, to make them equal number and length
        :param batch:
        :param max_seq_num:
        :param max_seq_len:
        :param pad_token:
        :return:
        """
        padded = []
        for ex in batch:
            padded.append([s[0] for s in ex] + [[pad_token] * max_seq_len] * (max_seq_num-len(ex)))
        return padded

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or lists of "phrases" (lists of str "words").
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[torch.LongTensor, torch.LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """
        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        if self.type and self.type == 'multiple':
            # print(self.type)
            # data: a list of phrases (list of words), [batch_size,seq_num,seq_len]
            batch_size = len(batch)
            max_seq_num = max([len(tgts) for tgts in batch])
            max_seq_len = max([len(p[0]) for e in batch for p in e])
            # make all examples have equal number of tgts, [batch_size, max_seq_num, max_seq_len]
            padded_data = self.pad_seqs(batch, max_seq_num, max_seq_len, self.base_field.pad_token)

            # flatten it to [batch_size*max_seq_num, max_seq_len]
            batch_by_feat = [seq for e in padded_data for seq in e]
            # base_data: [max_seq_len, batch_size*max_seq_num]
            base_data = self.base_field.process(batch_by_feat, device=device)
            # include_lengths is typically False (KeyphraseField is a target field)
            if self.base_field.include_lengths:
                # base_data: [max_seq_len, batch_size], lengths: batch_size
                base_data, lengths = base_data

            # feature is actually not supported
            feats = []
            levels = [base_data] + feats
            # data: [seq_len, batch_size*max_seq_num, len(self.fields)=1]
            data = torch.stack(levels, 2)
            # reshape it back to [seq_len, batch_size, max_seq_num, len(self.fields)=1]
            data = torch.reshape(data, shape=(-1, batch_size, max_seq_num, 1))

            if self.base_field.include_lengths:
                return data, lengths
            else:
                return data

        else:
            # [batch_size, seq_num=1, seq_len] -> [1, batch_size, seq_len]
            batch_by_feat = list(zip(*batch))
            base_data = self.base_field.process(batch_by_feat[0], device=device)
            if self.base_field.include_lengths:
                # base_data: [max_seq_len, batch_size], lengths: batch_size
                base_data, lengths = base_data

            feats = []
            levels = [base_data] + feats
            # data: seq_len x batch_size x len(self.fields) (usually only words, so num_feat=1)
            data = torch.stack(levels, 2)
            if self.base_field.include_lengths:
                return data, lengths
            else:
                return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """
        # if x is a list of strings (multiple keyphrases)
        if isinstance(x, list):
            return [[f.preprocess(x_) for _, f in self.fields] for x_ in x]
        else:
            return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def keyphrase_fields(**kwargs):
    """Create keyphrase fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        List[Tuple[str, KeyphraseField]]
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    # manually added in create_vocab()
    # sep = kwargs.get("sep", "<sep>")
    truncate = kwargs.get("truncate", None)
    lower = kwargs.get("lower", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim,
            lower = lower)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len, lower=lower)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = KeyphraseField(fields_[0][0], fields_[0][1])
    return field

