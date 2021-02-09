#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import json
import os
import time
import numpy as np
from itertools import count, zip_longest

import torch
import tqdm

from onmt.bin.train import prepare_fields_transforms
from onmt.constants import DefaultTokens
import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble

from onmt.inputters import KeyphraseDataset
from onmt.decoders import BARTDecoder
from onmt.encoders import PretrainedEncoder, BARTEncoder
from onmt.inputters.inputter import IterOnDevice
from onmt.train_single import _build_valid_iter

from onmt.translate.beam_search import BeamSearch, BeamSearchLM
from onmt.translate.greedy_search import GreedySearch, GreedySearchLM
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.utils.alignment import extract_alignment, build_align_pharaoh
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.constants import ModelTask


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None and opt.data_type != 'keyphrase':
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )
    fields, model, model_opt = load_test_model(opt)

    # (deprecated after dynamic data loading) added by @memray, ignore alignment field during testing for keyphrase task
    # if opt.data_type == 'keyphrase' and 'alignment' in fields:
    #     del fields['alignment']

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    if model_opt.model_task == ModelTask.LANGUAGE_MODEL:
        translator = GeneratorLM.from_opt(
            model,
            fields,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    else:
        translator = Translator.from_opt(
            model,
            fields,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    return translator


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Inference(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        tgt_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        fields,
        src_reader,
        tgt_reader,
        gpu=-1,
        n_best=1,
        min_length=0,
        max_length=100,
        ratio=0.0,
        beam_size=30,
        random_sampling_topk=1,
        random_sampling_temp=1,
        stepwise_penalty=None,
        dump_beam=False,
        block_ngram_repeat=0,
        ignore_when_blocking=frozenset(),
        replace_unk=False,
        tgt_prefix=False,
        phrase_table="",
        data_type="text",
        verbose=False,
        report_time=False,
        copy_attn=False,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
        seed=-1,
        kp_concat_type=None,
        model_kp_concat_type=None,
        beam_terminate=None,
        # **kwargs
        opt=None,
        model_opt=None
    ):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"]
        if hasattr(tgt_field, 'base_field'):
            tgt_field = tgt_field.base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = (
            torch.device("cuda", self._gpu)
            if self._use_cuda
            else torch.device("cpu")
        )

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking
        }
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError("replace_unk requires an attentional decoder.")
        self.tgt_prefix = tgt_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if (
            self.global_scorer.has_cov_pen
            and not self.model.decoder.attentional
        ):
            raise ValueError(
                "Coverage penalty requires an attentional decoder."
            )
        self.out_file = out_file
        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # added by @memray, to accommodate multiple targets
        self.kp_concat_type = kp_concat_type
        self.model_kp_concat_type = model_kp_concat_type
        # beam search termination condition
        self.beam_terminate = beam_terminate
        self.opt = opt
        self.model_opt = model_opt

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
        cls,
        model,
        fields,
        opt,
        model_opt,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
    ):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """
        # TODO: maybe add dynamic part
        cls.validate_task(model_opt.model_task)

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        # @memray a different tgt_reader for keyphrase
        if opt.data_type == 'keyphrase':
            data_type = 'keyphrase'
        else:
            data_type = 'text'
        tgt_reader = inputters.str2reader[data_type].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            tgt_prefix=opt.tgt_prefix,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            report_score=report_score,
            logger=logger,
            seed=opt.seed,
            kp_concat_type=opt.kp_concat_type,
            model_kp_concat_type=model_opt.kp_concat_type,
            beam_terminate=opt.beam_terminate,
            opt=opt,
            model_opt=model_opt
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(
        self,
        batch,
        memory_bank,
        src_lengths,
        src_vocabs,
        use_src_map,
        enc_states,
        batch_size,
        src,
        encoder_output=None
    ):
        if "tgt" in batch.__dict__:
            gs = self._score_target(
                batch,
                memory_bank,
                src_lengths,
                src_vocabs,
                batch.src_map if use_src_map else None,
                encoder_output=encoder_output)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs

    def translate(
        self,
        src,
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
        opt=None
    ):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        if batch_size is None:
            raise ValueError("batch_size must be set")

        if self.tgt_prefix and tgt is None:
            raise ValueError("Prefix should be feed to tgt if -tgt_prefix.")

        if self.data_type == 'text':
            use_dynamic_transform = False
            # original testing data pipeline
            src_data = {"reader": self.src_reader, "data": src}
            tgt_data = {"reader": self.tgt_reader, "data": tgt}
            _readers, _data = inputters.Dataset.config(
                [("src", src_data), ("tgt", tgt_data)]
            )

            # modified by @memray to accommodate keyphrase
            data = inputters.str2dataset[self.data_type](
                self.fields,
                readers=_readers,
                data=_data,
                sort_key=inputters.str2sortkey[self.data_type],
                filter_pred=self._filter_pred,
                data_format=opt.data_format,
                tgt_concat_type=opt.kp_concat_type
            )

            # @memray, as Dataset is only instantiated here, having to use this plugin setter
            if isinstance(data, KeyphraseDataset):
                data.kp_concat_type=self.kp_concat_type

            data_iter = inputters.OrderedIterator(
                dataset=data,
                device=self._dev,
                batch_size=batch_size,
                batch_size_fn=max_tok_len if batch_type == "tokens" else None,
                train=False,
                sort=False,
                sort_within_batch=False, #@memray: set False to keep the original order
                shuffle=False
            )
            src_vocabs = data.src_vocabs
            has_tgt = True if tgt is not None else False
        elif self.data_type == 'keyphrase':
            # dynamic data pipeline, lowercase must be inherited from model_opt
            use_dynamic_transform = True
            if hasattr(self.model_opt, 'lowercase'):
                setattr(opt, 'lowercase', self.model_opt.lowercase)
                setattr(self.opt, 'lowercase', self.model_opt.lowercase)

            _, transforms_cls = prepare_fields_transforms(opt)
            _data_iter = _build_valid_iter(opt, self.fields, transforms_cls)
            data_iter = IterOnDevice(_data_iter, device_id=self._gpu)
            data = None
            # src_vocabs is used in collapse_copy_scores and Translator.py
            src_vocabs = None
            has_tgt = True
        else:
            raise NotImplementedError('Currently only support data type=text/keyphrase.')

        xlation_builder = onmt.translate.TranslationBuilder(
            data,
            self.fields,
            self.n_best,
            self.replace_unk,
            has_tgt,
            self.phrase_table,
            use_dynamic_transform=use_dynamic_transform
        )
        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        num_examples = 0
        for batch_idx, batch in tqdm.tqdm(enumerate(data_iter), desc='Translating in batches'):
            _batch_size = batch.batch_size
            num_examples += _batch_size

            # @memray reshaping for dynamic preprocessing and keyphrase dataset
            if self.data_type == 'keyphrase':
                src, _ = (batch.src if isinstance(batch.src, tuple) else (batch.src, None))
                # for compatibility with previous versions, make src/tgt's dim to 3
                if len(src.shape) == 2:
                    src = src.unsqueeze(2)
                    tgt = batch.tgt.unsqueeze(2)
                    # required in generator
                    batch.src = src
                    batch.tgt = tgt
            # Output of dynamic batching is src.shape=[batch_size, length, num_feat]
            #   but OpenNMT expects [length, batch_size, num_feat]
            if batch.src[0].shape[0] == _batch_size:
                if isinstance(batch.src, tuple):
                    batch.src = (batch.src[0].permute([1, 0, 2]), batch.src[1])
                else:
                    batch.src = batch.src.permute([1, 0, 2])  # [src_len, B, 1]
                batch.tgt = batch.tgt.permute([1, 0, 2])  # [tgt_len, B, 1]

            batch_data = self.translate_batch(
                batch, src_vocabs, attn_debug
            )
            translations = xlation_builder.from_batch(batch_data)

            # @memray
            if self.data_type == "keyphrase":
                # post-process for one2seq outputs, split seq into individual phrases
                if self.model_kp_concat_type != 'one2one':
                    translations = self.segment_one2seq_trans(translations)
                # add statistics of kps(pred_num, beamstep_num etc.)
                translations = self.add_trans_stats(translations, self.kp_concat_type)

                # add copied flag
                if hasattr(self.fields['src'], 'base_field'):
                    vocab_size = len(self.fields['src'].base_field.vocab.itos)
                else:
                    vocab_size = len(self.fields['src'].vocab.itos)
                for t in translations:
                    t.add_copied_flags(vocab_size)

            for trans in translations:
                all_scores += [trans.pred_scores[: self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [
                    " ".join(pred) for pred in trans.pred_sents[: self.n_best]
                ]
                if self.report_align:
                    align_pharaohs = [
                        build_align_pharaoh(align)
                        for align in trans.word_aligns[: self.n_best]
                    ]
                    n_best_preds_align = [
                        " ".join(align) for align in align_pharaohs
                    ]
                    n_best_preds = [
                        pred + DefaultTokens.ALIGNMENT_SEPARATOR + align
                        for pred, align in zip(
                            n_best_preds, n_best_preds_align
                        )
                    ]
                all_predictions += [n_best_preds]

                if self.out_file:
                    import json
                    if self.data_type == "keyphrase":
                        self.out_file.write(json.dumps(trans.__dict__()) + '\n')
                        self.out_file.flush()
                    else:
                        self.out_file.write('\n'.join(n_best_preds) + '\n')
                        self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    if self.data_type == "keyphrase":
                        output = trans.log_kp(sent_number)
                    else:
                        output = trans.log(sent_number)

                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if align_debug:
                    tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score(
                "PRED", pred_score_total, pred_words_total
            )
            self._log(msg)
            if tgt is not None:
                msg = self._report_score(
                    "GOLD", gold_score_total, gold_words_total
                )
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log(
                "Average translation time (s): %f"
                % (total_time / len(all_predictions))
            )
            self._log(
                "Tokens per second: %f" % (pred_words_total / total_time)
            )

        if self.dump_beam:
            import json

            json.dump(
                self.translator.beam_accum,
                codecs.open(self.dump_beam, "w", "utf-8"),
            )
        return all_scores, all_predictions

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [
            best.tolist() for bests in predictions for best in bests
        ]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype,
            device=device,
        ).T
        bos_tensor = torch.full(
            [paded_tgt.size(0), 1], bos, dtype=dtype, device=device
        )
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1)
        )  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            avg_score = score_total / words_total
            ppl = np.exp(-score_total.item() / words_total)
            msg = "%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name,
                avg_score,
                name,
                ppl,
            )
        return msg

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        batch,
        src_vocabs,
        memory_lengths,
        src_map=None,
        step=None,
        batch_offset=None,
    ):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(
                dec_out.view(-1, dec_out.size(2)),
                attn.view(-1, attn.size(2)),
                src_map,
            )
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(-1, batch.batch_size, scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset,
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        return log_probs, attn

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        raise NotImplementedError

    def _score_target(
        self, batch, memory_bank, src_lengths, src_vocabs, src_map
    ):
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        batch,
        batch_size,
        src,
        src_lengths,
        src_vocabs,
        use_src_map,
        decode_strategy,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": gold_score,
        }

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions
            )
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        return results


class Translator(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.SEQ2SEQ:
            raise ValueError(
                f"Translator does not support task {task}."
                f" Tasks supported: {ModelTask.SEQ2SEQ}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        # (0) add BOS and padding to tgt prediction
        batch_tgt_idxs = self._align_pad_prediction(
            predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx
        )
        tgt_mask = (
            batch_tgt_idxs.eq(self._tgt_pad_idx)
            | batch_tgt_idxs.eq(self._tgt_eos_idx)
            | batch_tgt_idxs.eq(self._tgt_bos_idx)
        )

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(src_len, batch * n_best, nfeat)``
        src = tile(src, n_best, dim=1)
        enc_states = tile(enc_states, n_best, dim=1)
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, n_best, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, n_best, dim=1)
        src_lengths = tile(src_lengths, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1]  # exclude last target from inputs
        _, attns = self.model.decoder(
            dec_in, memory_bank, memory_lengths=src_lengths, with_align=True
        )

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(
            alignment_attn, prediction_mask, src_lengths, n_best
        )
        return alignement

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.beam_size == 1:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    batch_size=batch.batch_size,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    beam_terminate=self.beam_terminate
                )
            return self._translate_batch_with_strategy(
                batch, src_vocabs, decode_strategy)

    def _run_encoder(self, batch):
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )
        if src_lengths is None and hasattr(batch, 'src_length'):
            src_lengths = batch.src_length

        if isinstance(self.model.encoder, BARTEncoder):
            enc_states, memory_bank, encoder_output = self.model.encoder(src, src_lengths)
        elif isinstance(self.model.encoder, PretrainedEncoder):
            # for Transformer Decoder, only memory_bank is useful
            enc_states, memory_bank, ext_logits = self.model.encoder(src, batch.src_mask)
            encoder_output = None
        else:
            # enc_state is used for initializing OpenNMT decoders
            enc_states, memory_bank, src_lengths = self.model.encoder(
                src, src_lengths
            )
            encoder_output = None

        if src_lengths is None:
            assert not isinstance(
                memory_bank, tuple
            ), "Ensemble decoding only supported for text data"
            src_lengths = (
                torch.Tensor(batch.batch_size)
                .type_as(memory_bank)
                .long()
                .fill_(memory_bank.size(0))
            )
        return src, enc_states, memory_bank, src_lengths, encoder_output

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            batch,
            src_vocabs,
            memory_lengths,
            src_map=None,
            step=None,
            batch_offset=None,
            encoder_output=None,
            incremental_state=None,
            # @memray in general we just need the prediction of last word,
            #   but for _gold_score/_score_target we need to return the whole sequence
            last_word=True,
    ):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )
        # Inputs:
        #     decoder_in: [tgt_len, batch, nfeats]
        #     memory_bank: [src_len, batch, hidden]
        # Outputs:
        #    In case of inference tgt_len = 1, batch = beam x batch_size
        #        dec_out is in shape of [1, beam_size * batch_size, hidden]
        #        dec_attn is a dict which values are in shape of [1, beam_size * batch_size, src_len]
        #    In case of Gold Scoring tgt_len = actual length, batch = 1 x batch
        #        dec_out: [tgt_len, batch_size, dec_dim]
        #        dec_attn['std']: [tgt_len, batch_size, src_len]
        if isinstance(self.model.decoder, BARTDecoder):
            # BARTDecoder only uses decoder_in and encoder_output
            #    decoder_in.shape = (tgt_len, batch_size, 1)
            #    output=(tgt_len, batch_size, dec_dim), attn=(tgt_len, batch_size, src_len)
            dec_out, dec_attn = self.model.decoder(decoder_in, memory_bank,
                                                   memory_lengths=memory_lengths,
                                                   encoder_output=encoder_output,
                                                   incremental_state=incremental_state,
                                                   )
            # only preserve the output of last word, [tgt_len, B, D] -> [1, B, D]
            if last_word:
                dec_out = dec_out[-1, :, :].unsqueeze(0)
                if dec_attn:
                    dec_attn = {k:v[-1, :, :].unsqueeze(0) for k,v in dec_attn.items()}
        else:
            dec_out, dec_attn = self.model.decoder(
                decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
            )

        # Generator forward.
        if not self.copy_attn:
            if dec_attn and "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            assert dec_attn is not None, 'for copy generator, attention is required'
            attn = dec_attn["copy"]
            # for beam search, here we have scores [tgt_len x batch, vocab/cvocab] or [beam x batch, vocab/cvocab]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # gold_score [tgt_lenxbatch, tmp_vocab] -> [tgt_len, batch, tmp_vocab]???
            # beam search [beamxbatch, tmp_vocab] -> [batch, beam, tmp_vocab]
            if batch_offset is None:
                scores = scores.view(-1, batch.batch_size, scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            # Gold score: [batch, tgt_len, merged_vocab] -> [tgt_len, batch, merged_vocab]
            # Beam search: [batch, beam, merged_vocab] -> [1, batch*beam, merged_vocab]
            scores = scores.view(-1, decoder_in.size(1), scores.size(-1)) # to be compatible with BART. previous: scores.view(decoder_in.size(0), -1, scores.size(-1))
            # Gold score: [tgt_len, batch_size, vocab]
            # Beam search: [batch_size x beam_size, vocab]
            log_probs = scores.squeeze(0).log()

        # attn=(tgt_len, batch_size, dec_dim)
        return log_probs, attn

    def _translate_batch_with_strategy(
        self, batch, src_vocabs, decode_strategy
    ):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths, encoder_output = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        gold_score = self._gold_score(
            batch,
            memory_bank,
            src_lengths,
            src_vocabs,
            use_src_map,
            enc_states,
            batch_size,
            src,
        )

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        target_prefix = batch.tgt if self.tgt_prefix else None
        (
            fn_map_state,
            memory_bank,
            memory_lengths,
            src_map,
        ) = decode_strategy.initialize(
            memory_bank, src_lengths, src_map, target_prefix=target_prefix
        )
        incremental_state = {}
        if isinstance(self.model.decoder, BARTDecoder):
            new_order = torch.arange(batch_size).view(-1, 1).repeat(1, parallel_paths).view(-1)
            new_order = new_order.to(src.device).long()
            encoder_output = self.model.encoder.model.reorder_encoder_out(encoder_output, new_order)
        else:
            if fn_map_state is not None:
                self.model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            # print(step)
            if isinstance(self.model.decoder, BARTDecoder):
                # keep all predicted tokens, length is used for position embedding, (pred_len, batch_size * beam_size, 1)
                decoder_input = decode_strategy.alive_seq.permute(1, 0).unsqueeze(2)
            else:
                decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            # log_probs: [(batch_size x beam_size) , vocab] when 1 step or [ tgt_len, batch_size, vocab] when full sentence
            # attn: [1, (batch_size x beam_size), src_len]
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,
                encoder_output=encoder_output,
                incremental_state=incremental_state,
            )

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished(last_step=(step+1==decode_strategy.max_length))
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        x.index_select(1, select_indices) for x in memory_bank
                    )
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)
            if parallel_paths > 1 or any_finished:
                # @memray re-order decoder internal states based on the prev choice of beams
                # and drop the finished batched
                if isinstance(self.model.decoder, BARTDecoder):
                    self.model.decoder.model.reorder_incremental_state_scripting(incremental_state, select_indices)
                    encoder_output = self.model.encoder.model.reorder_encoder_out(encoder_output, select_indices)
                    # print('#select_indices=%d' % len(select_indices))
                    # print(select_indices)
                    # print('#finished=%d' % (torch.sum(decode_strategy.is_finished.int()).item()))
                    # print('incremental_state[0].shape=%s' % str(list(incremental_state.values())[0]['prev_key'].shape))
                    # print('encoder_output[0].shape=%s' % str(encoder_output[0].shape))
                else:
                    self.model.decoder.map_state(
                        lambda state, dim: state.index_select(dim, select_indices)
                    )


        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )

    def _score_target(
        self, batch, memory_bank, src_lengths, src_vocabs, src_map, encoder_output
    ):
        tgt = batch.tgt
        tgt_in = tgt[:-1] # trim EOS

        # score target sequence (ground-truth), log_probs=[tgt_len-1, batch_size, vocab], attn=[tgt_len-1, batch_size, src_len]
        log_probs, attn = self._decode_and_generate(
            tgt_in,
            memory_bank,
            batch,
            src_vocabs,
            memory_lengths=src_lengths,
            src_map=src_map,
            encoder_output=encoder_output,
            last_word=False
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:] # trim BOS
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores


    def _report_kpeval(self, src_path, tgt_path, pred_path):
        import subprocess
        path = os.path.abspath(__file__ + "/../../..")
        msg = subprocess.check_output(
            "python %s/tools/kp_eval.py -src %s -tgt %s -pred %s"
            % (path, src_path, tgt_path, pred_path),
            shell=True, stdin=self.out_file
        ).decode("utf-8").strip()
        return msg

    def add_trans_stats(self, trans, kp_concat_type):
        for tran in trans:
            if kp_concat_type == 'one2one':
                tran.unique_pred_num = len(tran.preds)
                tran.dup_pred_num = len(tran.preds)
                tran.beam_num = len(tran.preds)
                tran.beamstep_num = sum([len(t) for t in tran.preds])
            else:
                tran.beam_num = len(tran.ori_preds)
                tran.beamstep_num = sum([len(t) for t in tran.ori_preds])

        return trans

    def segment_one2seq_trans(self, trans):
        """
        For keyphrase generation tasks, one2seq models output sequences consisting of multiple phrases. Split them by delimiters and rerank them
        :param trans: a list of translations, length=batch_size, each translation contains multiple beams corresponding to one source text
        :return: a list of translations, each beam in each translation (multiple phrases delimited by <sep>) is a phrase
        """
        for tran in trans:
            dup_pred_tuples = []

            new_preds = []
            new_pred_sents = []
            new_pred_scores = []
            new_pred_counter = {}

            topseq_preds = []
            topseq_pred_sents = []
            topseq_pred_scores = []
            for sent_i in range(len(tran.pred_sents)):
                pred_sent = tran.pred_sents[sent_i]
                sep_indices = [i for i in range(len(pred_sent)) if pred_sent[i] == inputters.keyphrase_dataset.SEP_token]
                sep_indices = [-1] + sep_indices + [len(pred_sent)]

                for kp_i in range(len(sep_indices)-1):
                    start_idx = sep_indices[kp_i] + 1
                    end_idx = sep_indices[kp_i + 1]
                    new_kp = pred_sent[start_idx: end_idx]
                    new_kp_str = '_'.join(new_kp)

                    # keep all preds, even duplicate
                    dup_pred_tuples.append((tran.preds[sent_i][start_idx: end_idx],
                                      tran.pred_sents[sent_i][start_idx: end_idx],
                                      tran.pred_scores[sent_i]))

                    # skip duplicate
                    if new_kp_str in new_pred_counter:
                        new_pred_counter[new_kp_str] += 1
                        continue

                    # TODO, no account for attns and copies
                    new_pred_counter[new_kp_str] = 1
                    new_preds.append(tran.preds[sent_i][start_idx: end_idx])
                    new_pred_sents.append(tran.pred_sents[sent_i][start_idx: end_idx])
                    new_pred_scores.append(tran.pred_scores[sent_i])

                    # first beam (top-rank sequence)
                    if sent_i == 0:
                        topseq_preds.append(tran.preds[sent_i][start_idx: end_idx])
                        topseq_pred_sents.append(tran.pred_sents[sent_i][start_idx: end_idx])
                        topseq_pred_scores.append(tran.pred_scores[sent_i])

            # print('#(unique)/#(kp) = %d/%d' % (len(new_pred_counter), sum(new_pred_counter.values())))
            # print(new_pred_counter)

            # one2seq-specific stats
            tran.unique_pred_num = len(new_pred_counter)
            tran.dup_pred_num = sum(new_pred_counter.values())

            # still keep the original pred beams
            tran.ori_preds = tran.preds
            tran.ori_pred_sents = tran.pred_sents
            tran.ori_pred_scores = tran.pred_scores

            # segmented predictions from the top-score sequence
            tran.topseq_preds = topseq_preds
            tran.topseq_pred_sents = topseq_pred_sents
            tran.topseq_pred_scores = topseq_pred_scores

            # all segmented predictions
            tran.preds = new_preds
            tran.pred_sents = new_pred_sents
            tran.pred_scores = new_pred_scores

            tran.dup_pred_tuples = dup_pred_tuples

        return trans


class GeneratorLM(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.LANGUAGE_MODEL:
            raise ValueError(
                f"GeneratorLM does not support task {task}."
                f" Tasks supported: {ModelTask.LANGUAGE_MODEL}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        raise NotImplementedError

    def translate(
        self,
        src,
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        if batch_size != 1:
            warning_msg = ("GeneratorLM does not support batch_size != 1"
                           " nicely. You can remove this limitation here."
                           " With batch_size > 1 the end of each input is"
                           " repeated until the input is finished. Then"
                           " generation will start.")
            if self.logger:
                self.logger.info(warning_msg)
            else:
                os.write(1, warning_msg.encode("utf-8"))

        return super(GeneratorLM, self).translate(
            src,
            tgt,
            batch_size=1,
            batch_type=batch_type,
            attn_debug=attn_debug,
            align_debug=align_debug,
            phrase_table=phrase_table,
        )

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.beam_size == 1:
                decode_strategy = GreedySearchLM(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    batch_size=batch.batch_size,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearchLM(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                )
            return self._translate_batch_with_strategy(
                batch, src_vocabs, decode_strategy
            )

    def split_src_to_prevent_padding(self, src, src_lengths):
        min_len_batch = torch.min(src_lengths).item()
        target_prefix = None
        if min_len_batch > 0 and min_len_batch <= src.size(0):
            # hack [min_len_batch-1:] because expect <bos>
            target_prefix = (
                src[min_len_batch - 1:]
                if min_len_batch > 0 and min_len_batch <= src.size(0)
                else None
            )
            src = src[:min_len_batch]
            src_lengths[:] = min_len_batch
        return src, src_lengths, target_prefix

    def _translate_batch_with_strategy(
        self, batch, src_vocabs, decode_strategy
    ):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) split src into src and target_prefix to avoid padding.
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )

        src, src_lengths, target_prefix = self.split_src_to_prevent_padding(
            src, src_lengths
        )

        # (2) init decoder
        self.model.decoder.init_state(src, None, None)
        gold_score = self._gold_score(
            batch,
            None,
            src_lengths,
            src_vocabs,
            use_src_map,
            None,
            batch_size,
            src,
        )

        # (3) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        (
            fn_map_state,
            src,
            memory_lengths,
            src_map,
        ) = decode_strategy.initialize(
            src,
            src_lengths,
            src_map,
            target_prefix=target_prefix,
        )
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (4) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = (
                src
                if step == 0
                else decode_strategy.current_predictions.view(1, -1, 1)
            )

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                None,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths.clone(),
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,
            )

            if step == 0:
                log_probs = log_probs[-1]

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            memory_lengths += 1
            if any_finished:
                # Reorder states.
                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                # select indexes in model state/cache
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )

    def _score_target(
        self, batch, memory_bank, src_lengths, src_vocabs, src_map
    ):
        tgt = batch.tgt
        src, src_lengths = (
            batch.src if isinstance(batch.src, tuple) else (batch.src, None)
        )

        log_probs, attn = self._decode_and_generate(
            src,
            None,
            batch,
            src_vocabs,
            memory_lengths=src_lengths,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold_scores = log_probs.gather(2, tgt)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores
