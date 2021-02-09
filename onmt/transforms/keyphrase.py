# -*- coding: utf-8 -*-
import math
import numpy as np
import torch

from onmt.inputters.keyphrase_dataset import kpdict_parse_fn, KP_DATASET_FIELDS, KP_CONCAT_TYPES, obtain_sorted_indices, \
    parse_src_fn
from onmt.transforms import register_transform
from .transform import Transform


@register_transform(name='keyphrase')
class KeyphraseTransform(Transform):

    def __init__(self, opts):
        super().__init__(opts)
        self.kp_concat_type = opts.kp_concat_type
        self.max_target_phrases = opts.max_target_phrases
        self.lowercase = opts.lowercase

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
                  type=bool, default=False,
                  help="whether to apply lowercase to both source and target text.")

    def warm_up(self, vocabs):
        super().warm_up(None)
        if vocabs is None:
            self.bart_noise = None
            return
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
        if isinstance(ex_dict[keyword_field], str):
            tgt_kps = ex_dict[keyword_field].split(';')
        else:
            tgt_kps = ex_dict[keyword_field]
        if kp_concat_type == 'one2one':
            # sample one tgt from multiple tgts and use it as the only tgt
            rand_idx = np.random.randint(len(tgt_kps))
            tgt_str = tgt_kps[rand_idx]
        elif kp_concat_type in KP_CONCAT_TYPES:
            # generate one2seq training data points
            order = obtain_sorted_indices(src_str.lower().split(),
                                          [kp.lower().split() for kp in tgt_kps],
                                          sort_by=kp_concat_type)
            if max_target_phrases > 0 and len(order) > max_target_phrases:
                order = order[: max_target_phrases]
            tgt = [tgt_kps[idx] for idx in order]
        else:
            raise NotImplementedError('Unsupported target concatenation type ' + kp_concat_type)

        tgt_str = self.sep_token.join(tgt)
        if lowercase:
            return src_str.lower(), tgt_str.lower()
        return src_str, tgt_str

    def infer_dataset_type(self, example):
        dataset_type = None
        if 'question' in example:
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
        dataset_type = self.infer_dataset_type(example)
        src_str, tgt_str = self.kpdict_parse_fn(example, self.kp_concat_type,
                                           dataset_type=dataset_type, max_target_phrases=self.max_target_phrases,
                                           lowercase=self.lowercase)
        example['src'] = src_str
        example['tgt'] = tgt_str

        return example
