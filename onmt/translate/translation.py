""" Translation main class """
from __future__ import unicode_literals, print_function

import torch

from onmt.inputters import KeyphraseDataset
from onmt.inputters.text_dataset import TextMultiField
from onmt.utils.alignment import build_align_pharaoh


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table=""):
        self.data = data
        self.fields = fields
        # self._has_text_src = isinstance(dict(self.fields)["src"], TextMultiField)
        self._has_text_src = False # not tested
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field if hasattr(dict(self.fields)["tgt"], 'base_field') else dict(self.fields)["tgt"]
        pretrained_tokenizer = None
        if hasattr(tgt_field, 'pretrained_tokenizer'):
            pretrained_tokenizer = tgt_field.pretrained_tokenizer
        vocab = tgt_field.vocab
        tokens = []

        for tok in pred:
            if tok < len(vocab):
                if pretrained_tokenizer:
                    # directly use the pretrained tokenizer for decoding if applicable
                    token = pretrained_tokenizer.convert_ids_to_tokens([tok.item()])
                    tokens.append(token[0])
                else:
                    tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
        if pretrained_tokenizer:
            tokens = pretrained_tokenizer.convert_tokens_to_string(tokens).split()
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, align, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["alignment"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        if not any(align):  # when align is a empty nested list
            align = [None] * batch_size

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            if isinstance(batch.tgt, torch.Tensor) and len(batch.tgt.size()) == 3:
                src = batch.src[:, :, 0].index_select(1, perm) # src_len, batch_size
            else:
                src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None

        # if only one tgt: batch.tgt.dim=[max_seq_len, batch_size, 1]
        # if multiple tgts: batch.tgt.dim=[max_seq_len, batch_size, max_seq_num, 1]
        if self.has_tgt:
            if len(batch.tgt.size()) == 3:
                tgt = batch.tgt[:, :, 0].index_select(1, perm)
            else:
                tgt = batch.tgt[:, :, :, 0].index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            if self._has_text_src:
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src[0]
            else:
                src_vocab = None
                src_raw = None
            # @memray: len(preds[b]) can be smaller than n_best
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(min(self.n_best, len(preds[b])))]
            gold_sent = None
            if tgt is not None:
                if tgt.dim() == 2:
                    gold_sent = self._build_target_tokens(
                        src[:, b] if src is not None else None,
                        src_vocab, src_raw,
                        tgt[1:, b] if tgt is not None else None, None)
                else:
                    gold_sent = [self._build_target_tokens(
                        src[:, b] if src is not None else None,
                        src_vocab, src_raw,
                        tgt[1:, b, n], None)
                        for n in range(tgt.size(2))]
                    gold_sent = [s for s in gold_sent if all([t != '<blank>' for t in s])]
            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b], preds[b], align[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
        preds (List[LongTensor]): Original indices of predicted words, added by # @memray
        word_aligns (List[FloatTensor]): Words Alignment distribution for
            each translation.
    """

    __slots__ = ["src", "src_raw", "gold_sent", "gold_score", "word_aligns",
                 "attns", "copied_flags",
                 "unique_pred_num", "dup_pred_num", "beam_num", "beamstep_num",
                 "pred_sents", "pred_scores", "preds",
                 "ori_pred_sents", "ori_pred_scores", "ori_preds",
                 "topseq_pred_sents", "topseq_pred_scores", "topseq_preds",
                 "dup_pred_tuples"
                 ]
    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score, preds, word_aligns):
        self.src = src
        self.src_raw = src_raw
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.attns = attn
        self.preds = preds
        self.pred_sents = pred_sents
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.word_aligns = word_aligns

        self.dup_pred_num = 0 # number of all predicted phrases
        self.unique_pred_num = 0 # number of unique phrases
        self.beam_num = 0 # number of effective beams
        self.beamstep_num = 0 # number of effective beam search steps

        self.copied_flags = None
        self.ori_pred_sents = None
        self.ori_pred_scores = None
        self.ori_preds = None
        self.topseq_pred_sents = None
        self.topseq_pred_scores = None
        self.topseq_preds = None
        self.dup_pred_tuples = None

    def log(self, sent_number):
        """
        Log translation.
        """
        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.word_aligns is not None:
            pred_align = self.word_aligns[0]
            pred_align_pharaoh = build_align_pharaoh(pred_align)
            pred_align_sent = ' '.join(pred_align_pharaoh)
            msg.append("ALIGN: {}\n".format(pred_align_sent))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)

    def __dict__(self):
        """
        Added by @memray to facilitate exporting json
        :return:
        """
        ret = {slot: getattr(self, slot) for slot in self.__slots__}

        for slot in self.__slots__:
            if ret[slot] is None:
                continue
            if torch.cuda.is_available():
                if slot.endswith('src'):
                    ret[slot] = ret[slot].cpu().numpy().tolist()
                elif slot.endswith('pred_scores'):
                    ret[slot] = [t.cpu().numpy().tolist() for t in ret[slot]]
                elif slot.endswith('preds'):
                    ret[slot] = [t.cpu().numpy().tolist() for t in ret[slot]]
            else:
                if slot.endswith('src'):
                    ret[slot] = ret[slot].numpy().tolist()
                elif slot.endswith('pred_scores'):
                    ret[slot] = [t.numpy().tolist() for t in ret[slot]]
                elif slot.endswith('preds'):
                    ret[slot] = [t.numpy().tolist() for t in ret[slot]]

            if slot == "dup_pred_tuples":
                for tid, t in enumerate(ret["dup_pred_tuples"]):
                    if torch.cuda.is_available():
                        nt = (t[0].cpu().numpy().tolist() if isinstance(t[0], torch.Tensor) else t[0],
                              t[1],
                              t[2].cpu().item() if isinstance(t[2], torch.Tensor) else t[2])
                    else:
                        nt = (t[0].numpy().tolist() if isinstance(t[0], torch.Tensor) else t[0],
                              t[1],
                              t[2].item() if isinstance(t[2], torch.Tensor) else t[2])
                    ret["dup_pred_tuples"][tid] = nt

        return ret


    def log_kp(self, sent_number):
        """
        Log keyphrase generation.
        """

        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.gold_sent is not None:
            tgt_sent = '\n\t'.join([' '.join(tgt) for tgt in self.gold_sent])
            msg.append('GOLD {}: \n\t{}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for t_id, (score, sent, pred, copied_flag) in enumerate(zip(self.pred_scores, self.pred_sents, self.preds, self.copied_flags)):
                tmp_pred = pred.cpu().numpy().tolist() if torch.cuda.is_available() else pred.numpy().tolist()
                msg.append("[{}][{:.4f}] {} {} {}\n".format(t_id + 1, score, sent, tmp_pred, '[Copy!]' if any(copied_flag) else ''))

        return "".join(msg)

    def add_copied_flags(self, vocab_size):
        copied_flags = [pred.ge(vocab_size) for pred in self.preds]
        if torch.cuda.is_available():
            copied_flags = [t.cpu().numpy().tolist() for t in copied_flags]
        else:
            copied_flags = [t.numpy().tolist() for t in copied_flags]

        self.copied_flags = copied_flags