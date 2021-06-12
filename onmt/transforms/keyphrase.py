# -*- coding: utf-8 -*-
import argparse
import math
import random
import re
import string

import numpy as np
import torch

from onmt.inputters.keyphrase_dataset import KP_DATASET_FIELDS, KP_CONCAT_TYPES, obtain_sorted_indices, \
    parse_src_fn
from onmt.keyphrase import utils
from onmt.transforms import register_transform
from .transform import Transform

import spacy
spacy_nlp = spacy.load('en_core_web_sm')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@register_transform(name='keyphrase')
class KeyphraseTransform(Transform):

    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self._parse_opts()

    def _parse_opts(self):
        super()._parse_opts()
        self.kp_concat_type = self.opts.kp_concat_type
        self.max_target_phrases = self.opts.max_target_phrases
        self.lowercase = self.opts.lowercase
        self.use_given_inputs = self.opts.use_given_inputs

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to BART."""
        group = parser.add_argument_group("Transform/keyphrase")
        group.add('--kp_concat_type', '-kp_concat_type', default=None,
                   choices=KP_CONCAT_TYPES,
                   help="""Format of targets for model to learn/output, 'multiple' is used during test phase.
                   Note that the names have been changed after the empirical paper, e.g. verbatim_append is changed to pres_abs. 
                   """)
        group.add("--max_target_phrases", "-max_target_phrases",
                  type=int, default=-1,
                  help="Number of phrases allowed on the target side, -1 denotes no limit.")
        group.add("--lowercase", "-lowercase",
                  type=str2bool, default=False,
                  help="whether to apply lowercase to both source and target text.")
        group.add("--use_given_inputs", "-use_given_inputs",
                  type=str2bool, default=False,
                  help="Only designated to ease the out-of-the-box inference,"
                       "using src and tgt that are directly given in ex['src'] and ex['tgt'] rather than processing them on-the-fly.")

    def warm_up(self, vocabs):
        super().warm_up(None)
        self.vocabs = vocabs
        # for now it is hard-coded, vocabs does not contain useful info if pre-trained tokenizer is used
        self.sep_token = '<sep>'

    def kpdict_parse_fn(self, ex_dict, kp_concat_type, dataset_type='scipaper',
                        max_target_phrases=-1, lowercase=False):
        """
        Similar to the function used on
        :param ex_dict:
        :param kp_concat_type:
        :param dataset_type:
        :param max_target_phrases:
        :param lowercase:
        :return:
        """
        assert dataset_type in KP_DATASET_FIELDS
        title_field, text_field, keyword_field, category_field = KP_DATASET_FIELDS[dataset_type]

        src_str = parse_src_fn(ex_dict, title_field, text_field)
        # make sure each phrase is a separate text string, not a list nor concatenated multiple phrases
        if isinstance(ex_dict[keyword_field], str):
            tgt_kps = ex_dict[keyword_field].split(';')
        elif isinstance(ex_dict[keyword_field], list) and len(ex_dict[keyword_field]) > 0 \
            and isinstance(ex_dict[keyword_field][0], list):
            tgt_kps = [' '.join(p) for p in ex_dict[keyword_field]]
        else:
            tgt_kps = ex_dict[keyword_field]
        if kp_concat_type == 'one2one':
            # sample one tgt from multiple tgts and use it as the only tgt
            rand_idx = np.random.randint(len(tgt_kps))
            tgt_str = tgt_kps[rand_idx]
        elif kp_concat_type in KP_CONCAT_TYPES:
            # (deprecated, extremely slow)generate one2seq training data points, use spacy tokenization
            # src_seq = [t.text.lower() for t in spacy_nlp(src_str, disable=["textcat"])]
            # tgt_seqs = [[t.text.lower() for t in spacy_nlp(p, disable=["textcat"])] for p in tgt_kps]
            src_seq = [t for t in re.split(r'\W', src_str) if len(t) > 0]
            tgt_seqs = [[t for t in re.split(r'\W', p) if len(t) > 0] for p in tgt_kps]

            if len(tgt_seqs) > 0:
                order = obtain_sorted_indices(src_seq, tgt_seqs,
                                              sort_by=kp_concat_type)
                if max_target_phrases > 0 and len(order) > max_target_phrases:
                    order = order[: max_target_phrases]
                tgt = [tgt_kps[idx] for idx in order]
                tgt_str = self.sep_token.join(tgt)
            else:
                tgt_str = ''
        else:
            raise NotImplementedError('Unsupported target concatenation type ' + kp_concat_type)

        if lowercase:
            return src_str.lower(), tgt_str.lower()
        return src_str, tgt_str

    def infer_dataset_type(self, example):
        dataset_type = None
        if 'dataset_type' in example:
            dataset_type = example['dataset_type']
        elif 'question' in example:
            dataset_type = 'qa'
        elif 'url' in example:
            dataset_type = 'webpage'
        elif 'date' in example:
            dataset_type = 'news'
        elif 'abstract' in example:
            dataset_type = 'scipaper'

        assert dataset_type is not None, 'Fail to detect the data type of the given input file.' \
                                         'Accecpted values:' + KP_DATASET_FIELDS.keys()
        return dataset_type

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """
        Source text: concatenating title and body text.
        Target text: concatenating phrases according to the given phrase order.
        """
        if self.use_given_inputs and 'src' in example and 'tgt' in example and example['src'] and example['tgt']:
            print('WARNING: using src and tgt that are directly given rather than processed on-the-fly.\n '
                  'This is only designated to ease the out-of-the-box inference. Ensure this behavior is wanted.')
            return example

        dataset_type = self.infer_dataset_type(example)
        src_str, tgt_str = self.kpdict_parse_fn(example, self.kp_concat_type,
                                                dataset_type=dataset_type, max_target_phrases=self.max_target_phrases,
                                                lowercase=self.lowercase)
        example['src'] = src_str
        example['tgt'] = tgt_str
        example['src_str'] = src_str
        example['tgt_str'] = tgt_str

        return example


@register_transform(name='kp_replace_target')
class KeyphraseReplaceTargetTransform(Transform):

    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self._parse_opts()

    def _parse_opts(self):
        super()._parse_opts()
        self.kp_concat_type = self.opts.kp_concat_type
        self.max_target_phrases = self.opts.max_target_phrases
        self.max_phrase_len = self.opts.max_phrase_len
        self.lowercase = self.opts.lowercase
        self.use_given_inputs = self.opts.use_given_inputs
        self.label_sample_ratio = self.opts.label_sample_ratio
        self.seed = self.opts.seed

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to BART."""
        group = parser.add_argument_group("Transform/kp_replace_target")
        group.add("--label_sample_ratio", type=float, nargs='+', help="Sampling proportion of labels from each label file.")

    def warm_up(self, vocabs):
        super().warm_up(None)

    def maybe_replace_target(self, example, label_sample_ratio,
                             max_target_phrases, max_phrase_len=-1,
                             add_control_prefix_prob=0.0,
                             fix_target_number=False, allow_duplicate=False,
                             sep_token='<sep>', seed=0):
        '''
        If additional target label sets are given, we replace example['target'] with new labels
        :param example:
        :param label_sample_ratio: sampling ratio of each extra label set
        :param max_target_phrases:
        :param add_control_prefix_prob: if given, we append the number of phrases as a prefix to source string
        :param fix_target_number: if True, target always contains `max_target_phrases` phrases, otherwise it's sampled in (0, max_target_phrases]
        :param allow_duplicate: if True, target can contain duplicate phrases, otherwise duplicate phrases are removed
        :param seed:
        :return:
        '''
        if not label_sample_ratio:  # label set is not given, directly return
            return example

        with utils.numpy_seed(seed):
            tgts = []
            for labelset_id, ratio in enumerate(label_sample_ratio):
                candicate_labels = example['target%d' % labelset_id]
                # ensure each phrase has less than 70 characters and max_phrase_len words
                if max_phrase_len > 0:
                    candicate_labels = [p for p in candicate_labels
                                        if len(p) < 70 and len(re.findall(r"\w+|[^\w\s]", p, re.UNICODE)) <= max_phrase_len]
                # remove punctuations
                candicate_labels = [p.strip() for p in candicate_labels if len(p.strip()) > 0]

                # determine number of phrases to sample
                if len(candicate_labels) == 0:
                    continue
                if max_target_phrases < 0:
                    num_to_sample = len(candicate_labels)
                else:
                    num_to_sample = min(len(candicate_labels), int(ratio * max_target_phrases))
                if num_to_sample == 0:
                    num_to_sample = 1

                tgts.extend(np.random.choice(candicate_labels, num_to_sample, replace=False))

            # deduplicate
            if not allow_duplicate:
                tgts = list(set(tgts))

            if len(tgts) == 0:
                return '', ''

            # shuffle order and randomize target size

            # print(len(tgts))
            # print(tgts)

            np.random.shuffle(tgts)
            if not fix_target_number:
                tgts = np.random.choice(tgts, size=np.random.randint(len(tgts)) + 1, replace=False).tolist()

            # print(len(tgts))
            # print(tgts)

            tgt_str = sep_token.join(tgts)

            # add control prefix (number of phrases to output)
            if add_control_prefix_prob > 0.0 and np.random.rand() < add_control_prefix_prob:
                prefix_str = '<mixed><number>%d<s>' % (len(tgts))
                src_str = prefix_str + example['src']
            else:
                src_str = example['src']

        return src_str, tgt_str

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """
        Source text: concatenating title and body text.
        Target text: concatenating phrases according to the given phrase order.
        """
        src_str, tgt_str = self.maybe_replace_target(example,
                                                     label_sample_ratio=self.label_sample_ratio,
                                                     max_target_phrases=self.max_target_phrases,
                                                     max_phrase_len=self.max_phrase_len,
                                                     fix_target_number=False, allow_duplicate=False,
                                                     sep_token='<sep>',
                                                     seed=self.seed
                                                     )
        # occasionally candidate tgt is empty, skip
        if len(tgt_str) > 0:
            example['src'] = src_str
            example['tgt'] = tgt_str
            example['src_str'] = src_str
            example['tgt_str'] = tgt_str

        return example


@register_transform(name='kp_random_span')
class KeyphraseRandomSpanTargetTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self._parse_opts()

    def _parse_opts(self):
        super()._parse_opts()
        self.seed = self.opts.seed
        self.max_phrase_len = self.opts.max_phrase_len
        self.max_target_phrases = self.opts.max_target_phrases
        self.lowercase = self.opts.lowercase

    @classmethod
    def add_options(cls, parser):
        group = parser.add_argument_group("Transform/keyphrase_random_span")
        group.add("--max_phrase_len", "-max_phrase_len",
                  type=int, default=-1,
                  help="Max length of target phrases.")

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    def warm_up(self, vocabs):
        super().warm_up(None)

    def random_span_parse_fn(self, ex, sep_token,
                    max_target_phrases=-1,
                    span_len_opts=None,
                    len_distrib=None,
                    lowercase=False,
                    seed=0):
        """
        :param ex:
        :param sep_token:
        :param max_target_phrases:
        :param lowercase:
        :return:
        """
        assert max_target_phrases > 0, 'max_target_phrases must be a positive integer'
        src_text = ex['src']

        with utils.numpy_seed(seed):
            # mask random spans
            src_tokens = src_text.split()

            span_lens = []
            num_span = np.random.random_integers(max_target_phrases)
            for i in range(num_span):
                span_len = np.random.choice(span_len_opts, p=len_distrib)
                span_lens.append(span_len)

            span_lens = sorted(span_lens, reverse=True)  # ensure larger spans get processed first

            spans = []
            uncovered_spans = [(0, len(src_tokens))]
            for span_len in span_lens:
                candicate_spans, noncandicate_spans = [], []
                for s in uncovered_spans:
                    if s[1] - s[0] >= span_len:
                        candicate_spans.append(s)
                    else:
                        noncandicate_spans.append(s)

                if len(candicate_spans) == 0:
                    # not possible to fit this span
                    continue
                candicate_span_id = random.choice(range(len(candicate_spans)))
                candicate_span = candicate_spans[candicate_span_id]
                candicate_span_len = candicate_span[1] - candicate_span[0]

                # sample a span start given the candidate
                span_start_offset = random.randint(0, candicate_span_len - span_len + 1)
                span_left = candicate_span[0] + span_start_offset
                spans.append((span_left, span_left + span_len))

                # maintain the new candidate lists
                if span_start_offset == 0:
                    leftover_spans = [(candicate_span[0] + span_len, candicate_span[1] + 1)]
                elif span_start_offset == candicate_span_len - span_len:
                    leftover_spans = [(candicate_span[0], candicate_span[1] - span_len)]
                else:
                    leftover_spans = [(candicate_span[0], span_left), (span_left + span_len, candicate_span[1] + 1)]

                uncovered_spans = noncandicate_spans + leftover_spans

            spans = sorted(spans, key=lambda x: x[0], reverse=False)
            infill_phrases = [' '.join(src_tokens[s[0]: s[1]]) for s in spans]

        tgt_text = sep_token.join(infill_phrases)

        if lowercase:
            return src_text.lower(), tgt_text.lower()

        return src_text, tgt_text

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """
        Source text: concatenating title and body text.
        Target text: concatenating phrases according to the given phrase order.
        """
        span_lens = list(range(1, self.max_phrase_len + 1))
        geometric_p = 0.2
        len_distrib = [geometric_p * (1 - geometric_p) ** (i - 1) for i in
                       range(1, self.max_phrase_len + 1)] if geometric_p >= 0 else None
        len_distrib = [x / (sum(len_distrib)) for x in len_distrib]

        src_str, tgt_str = self.random_span_parse_fn(example,
                                                     sep_token='<sep>',
                                                     max_target_phrases=self.max_target_phrases,
                                                     span_len_opts=span_lens, len_distrib=len_distrib,
                                                     lowercase=self.lowercase,
                                                     seed=self.seed
                                                     )
        example['src'] = src_str
        example['tgt'] = tgt_str
        example['src_str'] = src_str
        example['tgt_str'] = tgt_str

        return example


@register_transform(name='add_control_prefix')
class ControlPrefixTransform(Transform):

    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self._parse_opts()

    def _parse_opts(self):
        super()._parse_opts()
        self.src_control_prefix = self.opts.src_control_prefix
        self.tgt_control_prefix = self.opts.tgt_control_prefix


    @classmethod
    def add_options(cls, parser):
        """Available options relate to BART."""
        group = parser.add_argument_group("Transform/keyphrase")
        group.add("--src_control_prefix", "-src_control_prefix",
                  type=str, default='', help="It will be appended to the source text to control the output.")
        group.add("--tgt_control_prefix", "-tgt_control_prefix",
                  type=str, default='', help="It will be appended to the target text to control the output.")

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """
        Source text: concatenating title and body text.
        Target text: concatenating phrases according to the given phrase order.
        """
        # if control-prefix is given, prepend it
        if 'src_control_prefix' in example:
            src_control_prefix = example['src_control_prefix']
        elif self.src_control_prefix:
            src_control_prefix = self.src_control_prefix
        else:
            src_control_prefix = ''

        if 'tgt_control_prefix' in example:
            tgt_control_prefix = example['tgt_control_prefix']
        elif self.tgt_control_prefix:
            tgt_control_prefix = self.tgt_control_prefix
        else:
            tgt_control_prefix = ''

        example['src'] = src_control_prefix + example['src']
        example['tgt'] = tgt_control_prefix + example['tgt']

        example['src_str'] = example['src']
        example['tgt_str'] = example['tgt']

        return example


def findBalanced(text, openDelim=['[['], closeDelim=[']]']):
    """
    Assuming that text contains a properly balanced expression using
    :param openDelim: as opening delimiters and
    :param closeDelim: as closing delimiters.
    :return: an iterator producing pairs (start, end) of start and end
    positions in text containing a balanced expression.
    """
    openPat = '|'.join([re.escape(x) for x in openDelim])
    # pattern for delimiters expected after each opening delimiter
    afterPat = {o: re.compile(openPat + '|' + c, re.DOTALL) for o, c in zip(openDelim, closeDelim)}
    stack = []
    start = 0
    cur = 0
    # end = len(text)
    startSet = False
    startPat = re.compile(openPat)
    nextPat = startPat
    while True:
        next = nextPat.search(text, cur)
        if not next:
            return
        if not startSet:
            start = next.start()
            startSet = True
        delim = next.group(0)
        if delim in openDelim:
            stack.append(delim)
            nextPat = afterPat[delim]
        else:
            opening = stack.pop()
            # assert opening == openDelim[closeDelim.index(next.group(0))]
            if stack:
                nextPat = afterPat[stack[-1]]
            else:
                yield start, next.end()
                nextPat = startPat
                start = next.end()
                startSet = False
        cur = next.end()


def replaceInternalLinks(text, return_anchor_text=False):
    """
    Replaces internal links of the form:
    [[title |...|label]]trail

    with title concatenated with trail, when present, e.g. 's' for plural.

    See https://www.mediawiki.org/wiki/Help:Links#Internal_links
    """
    # call this after removal of external links, so we need not worry about
    # triple closing ]]].
    cur = 0
    res = ''
    phrase_list = []
    for s, e in findBalanced(text):
        m = tailRE.match(text, e)
        if m:
            trail = m.group(0)
            end = m.end()
        else:
            trail = ''
            end = e
        inner = text[s + 2:e - 2]
        # find first |
        pipe = inner.find('|')
        if pipe < 0:
            title = inner
            label = title
        else:
            title = inner[:pipe].rstrip()
            # find last |
            curp = pipe + 1
            for s1, e1 in findBalanced(inner):
                last = inner.rfind('|', curp, s1)
                if last >= 0:
                    pipe = last  # advance
                curp = e1
            label = inner[pipe + 1:].strip()

        # phrase_list.append(title.strip())
        phrase_list.append(label.strip())
        res += text[cur:s] + label + trail
        cur = end
    if return_anchor_text:
        return res + text[cur:], phrase_list
    else:
        return res + text[cur:]

bold_italic = re.compile(r"'''''(.*?)'''''")
bold = re.compile(r"'''(.*?)'''")
italic_quote = re.compile(r"''\"([^\"]*?)\"''")
italic = re.compile(r"''(.*?)''")
quote_quote = re.compile(r'""([^"]*?)""')
tailRE = re.compile('\w+')

def extract_phrases(text):
    # Extract bold/italic text and internal links

    # Extract bold/anchor texts
    font_phrases = bold_italic.findall(text)
    font_phrases += bold.findall(text)
    font_phrases += italic_quote.findall(text)
    font_phrases += italic.findall(text)
    font_phrases += quote_quote.findall(text)
    font_phrases = [p.strip('\',\"') for p in font_phrases]
    font_phrases = list(set(font_phrases))

    # Handle bold/italic/quote
    text = bold_italic.sub(r'\1', text)
    text = bold.sub(r'\1', text)
    text = italic_quote.sub(r'"\1"', text)
    text = italic.sub(r'"\1"', text)
    text = quote_quote.sub(r'"\1"', text)
    # replace internal links
    text, anchor_phrases = replaceInternalLinks(text, return_anchor_text=True)
    anchor_phrases = [p.strip('\',\"') for p in anchor_phrases]
    anchor_phrases = list(set(anchor_phrases))

    return text, font_phrases, anchor_phrases


@register_transform(name='wiki_phrase')
class WikiPhraseTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self._parse_opts()

    def _parse_opts(self):
        super()._parse_opts()
        self.kp_concat_type = self.opts.kp_concat_type
        self.max_target_phrases = self.opts.max_target_phrases
        self.max_phrase_len = self.opts.max_phrase_len
        self.lowercase = self.opts.lowercase
        self.seed = self.opts.seed
        self.phrase_corr_rate = self.opts.phrase_corr_rate
        self.random_span_rate = self.opts.random_span_rate
        # span distribution follows SpanBERT (https://arxiv.org/pdf/1907.10529.pdf)
        self.span_len_opts = list(range(1, 8 + 1))
        geometric_p = 0.2
        len_distrib = [geometric_p * (1 - geometric_p) ** (i - 1) for i in
                       range(1, 8 + 1)] if geometric_p >= 0 else None
        self.len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
        self.sep_token = '<sep>'

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to BART."""
        group = parser.add_argument_group("Transform/keyphrase")
        group.add_argument("--phrase_corr_rate", default=0.0, type=float,
                            help='.')
        group.add_argument("--random_span_rate", default=0.0, type=float,
                            help='.')

    def wiki_ex_parse_fn(self,
                         ex_dict, sep_token,
                         max_phrase_len=8,
                         max_target_phrases=-1,
                         phrase_corr_rate=0.0,
                         random_span_rate=0.0,
                         lowercase=False,
                         seed=0):
        """
        max_tgt_len=max_phrase_len*max_target_phrases + src_len*random_span_rate = 6*16+512*5%=96+25.6=121.6
        masked_word=6*8*0.1+512*5%=30.4 (30.4/512=5.9%)
        :param ex_dict:
        :param max_phrase_len:
        :param max_target_phrases:
        :param phrase_corr_rate: replace p% * num_present_phrase present phrases from src_text with <present>
        :param random_span_rate: replace p% * num_word spans from src_text with <mask>
        :param lowercase:
        :return:
        """
        assert max_target_phrases > 0, 'max_target_phrases must be a positive integer'
        text_field = 'text'

        src_text = ex_dict[text_field]
        src_text, font_phrases, anchor_phrases = extract_phrases(src_text)

        pres_phrases = set(font_phrases + anchor_phrases)
        header_phrases = [ex_dict['title']] + ex_dict['headers']
        category_phrases = ex_dict['categories']
        seealso_phrases = ex_dict['seealso']

        if max_phrase_len:
            pres_phrases = [p for p in pres_phrases if len(p.split()) <= max_phrase_len]
            header_phrases = [p for p in header_phrases if len(p.split()) <= max_phrase_len]
            category_phrases = [p for p in category_phrases if len(p.split()) <= max_phrase_len]
            seealso_phrases = [p for p in seealso_phrases if len(p.split()) <= max_phrase_len]

        with utils.numpy_seed(seed):
            # present phrases
            if max_target_phrases > 0 and len(pres_phrases) > max_target_phrases / 2:
                pres_phrases = np.random.choice(pres_phrases, int(max_target_phrases / 2), replace=False).tolist()

            num_pres = len(pres_phrases)
            num_header = len(header_phrases)
            num_cat = len(category_phrases)
            num_seealso = len(seealso_phrases)

            # absent phrases
            abs_phrases = header_phrases + category_phrases + seealso_phrases
            if max_target_phrases > 0 and len(abs_phrases) > max_target_phrases / 2:
                num_cat = min(len(category_phrases), random.randint(0, int(max_target_phrases / 2 - len(header_phrases))))
                num_seealso = min(len(seealso_phrases), int(max_target_phrases / 2) - len(header_phrases) - num_cat)
                abs_phrases = header_phrases \
                              + np.random.choice(category_phrases, num_cat, replace=False).tolist()\
                              + np.random.choice(seealso_phrases, num_seealso, replace=False).tolist()

            # mask random spans
            num_infill = 0
            if random_span_rate > 0.0:
                src_tokens = src_text.split()
                num_word_left = max(1, int(random_span_rate * len(src_tokens)))

                span_lens = []
                while num_word_left > 0:
                    span_len = np.random.choice(self.span_len_opts, p=self.len_distrib).tolist()
                    if span_len <= num_word_left:
                        span_lens.append(span_len)
                    else:
                        span_lens.append(num_word_left)
                    num_word_left -= span_len
                span_lens = sorted(span_lens, reverse=True) # ensure larger spans get processed first

                spans = []
                uncovered_spans = [(0, len(src_tokens))]
                for span_len in span_lens:
                    candicate_spans, noncandicate_spans = [], []
                    for s in uncovered_spans:
                        if s[1] - s[0] >= span_len:
                            candicate_spans.append(s)
                        else:
                            noncandicate_spans.append(s)

                    if len(candicate_spans) == 0:
                        # not possible to fit this span
                        continue
                    candicate_span_id = random.choice(range(len(candicate_spans)))
                    candicate_span = candicate_spans[candicate_span_id]
                    candicate_span_len = candicate_span[1] - candicate_span[0]

                    # sample a span start given the candidate
                    span_start_offset = random.randint(0, candicate_span_len - span_len + 1)
                    span_left = candicate_span[0] + span_start_offset
                    spans.append((span_left, span_left + span_len))

                    # maintain the new candidate lists
                    if span_start_offset == 0:
                        leftover_spans = [(candicate_span[0] + span_len, candicate_span[1] + 1)]
                    elif span_start_offset == candicate_span_len - span_len:
                        leftover_spans = [(candicate_span[0], candicate_span[1] - span_len)]
                    else:
                        leftover_spans = [(candicate_span[0], span_left), (span_left + span_len, candicate_span[1] + 1)]

                    uncovered_spans = noncandicate_spans + leftover_spans

                spans = sorted(spans, key=lambda x: x[0], reverse=False)
                masked_src_tokens = []
                prev_span_end = 0
                for s in spans:
                    masked_src_tokens.extend(src_tokens[prev_span_end: s[0]])
                    masked_src_tokens.append('<infill>')
                    prev_span_end = s[1]
                masked_src_tokens.extend(src_tokens[prev_span_end:])

                infill_phrases = [' '.join(src_tokens[s[0]: s[1]]) for s in spans]
                num_infill = len(infill_phrases)
                src_text = ' '.join(masked_src_tokens)

            # mask random present phrases
            if phrase_corr_rate > 0.0 and len(pres_phrases) > 0:
                num_mask_kp = min(1, int(len(pres_phrases) * phrase_corr_rate))
                mask_pres_phrases = np.random.choice(pres_phrases, num_mask_kp, replace=False).tolist()
                for p in mask_pres_phrases:
                    src_text = re.sub(p, '<present>', src_text, flags=re.IGNORECASE)

        prefix_str = '<present>%d<header>%d<category>%d<seealso>%d<infill>%d<s>' \
                     % (num_pres, num_header, num_cat, num_seealso, num_infill)

        src_text = prefix_str + src_text
        tgt_text = sep_token.join(pres_phrases + abs_phrases + infill_phrases)

        if lowercase:
            return src_text.lower(), tgt_text.lower()
        return src_text, tgt_text


    def apply(self, example, is_train=False, stats=None, **kwargs):
        """
        Source text: concatenating title and body text.
        Target text: concatenating phrases according to the given phrase order.
        """
        src_str, tgt_str = self.wiki_ex_parse_fn(example, sep_token=self.sep_token,
                                                 max_phrase_len=self.max_phrase_len,
                                                 max_target_phrases=self.max_target_phrases,
                                                 phrase_corr_rate=self.phrase_corr_rate,
                                                 random_span_rate=self.random_span_rate,
                                                 lowercase=self.lowercase,
                                                 seed=self.seed)

        example['src'] = src_str
        example['tgt'] = tgt_str
        example['src_str'] = src_str
        example['tgt_str'] = tgt_str

        return example
