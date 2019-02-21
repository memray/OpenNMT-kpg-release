# -*- coding: utf-8 -*-
import json
import re
from functools import partial

import six
import torch
import numpy as np
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example

from onmt.inputters.dataset_base import _join_dicts, _dynamic_dict

SEP_token = "<sep>"

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
                 filter_pred=None):
        self.tgt_type = None
        # concatenate multiple tgt sequences with <sep> or keep them separate as a list of seqs (2D tensor)
        self.concat_tgt = False

        self.sort_key = sort_key
        # if sort_key == None:
        #     sort_key = kp_sort_key
        # else:
        #     self.sort_key = sort_key

        # will be specified before training, one of [one2one, original, random, verbatim]
        can_copy = 'src_map' in fields and 'alignment' in fields

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                src_field = fields['src'][0][1]
                tgt_field = fields['tgt'][0][1]
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: v for k, v in fields.items() if k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # the dataset's self.fields should have the same attributes as examples
        fields = dict(chain.from_iterable(ex_fields.values()))

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
        for i, json_line in enumerate(sequences):
            json_line = json_line.decode("utf-8")
            json_dict = json.loads(json_line)
            # Note tgt could be a list of strings
            seq = json_dict[side]

            # torchtext field only takes numeric features
            id = json_dict['id']

            try:
                if id.rfind('_') != -1:
                    id = id[id.rfind('_') + 1:]
                id = int(id)
            except ValueError:
                # if not convertible, use indices as id
                id = i

            yield {side: seq, "indices": i, 'id': id}


def process_multiple_tgts(big_batch, tgt_type):
    assert tgt_type in ['one2one', 'no_sort', 'random', 'verbatim', 'alphabetical']
    np.random.seed(2333)

    new_batch = []
    for ex in big_batch:
        if len(ex.tgt) == 0:
            continue

        if tgt_type == 'one2one':
            # sample one tgt from multiple tgts and use it as the only tgt
            rand_idx = np.random.randint(len(ex.tgt))
            tgt = ex.tgt[rand_idx]
            alignment = ex.alignment[rand_idx] if hasattr(ex, "alignment") else None
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
        else:
            raise NotImplementedError

        # print(ex.tgt)
        # print(tgt)
        ex.tgt = tgt
        if hasattr(ex, "alignment"):
            assert len(tgt[0]) + 2 == alignment.size()[0]
            ex.alignment = alignment

        new_batch.append(ex)

    return new_batch


def kp_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    # if tgt is available, order by number of phrases in tgt first, then src length
    if hasattr(ex, "tgt"):
        return len(ex.tgt), len(ex.src[0])
    return len(ex.src[0])


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
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text)))

    # TODO: should be moved out
    # replace the digit terms with <digit>
    # tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens


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
    tokens = copyseq_tokenize(string)
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

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
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
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = []
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
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


def keyphrase_fields(base_name, **kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        List[Tuple[str, TextMultiField]]
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
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
    return [(base_name, field)]
