#!/usr/bin/env python
"""Training on a single process."""
import os
import shutil

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple, make_tgt, reload_news_fields
from onmt.inputters.news_dataset import load_pretrained_tokenizer
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from torchtext.data import Field, RawField

def _check_save_model_path(opt):
    if not hasattr(opt, 'save_model') or opt.save_model is None:
        if hasattr(opt, 'exp_dir'):
            setattr(opt, 'save_model', opt.exp_dir+'/models/model')
        else:
            raise Exception('Neither exp_dir nor save_model is not given!')

    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

    if not hasattr(opt, 'log_file') or opt.log_file is None or opt.log_file=='':
        if hasattr(opt, 'log_file'):
            setattr(opt, 'log_file', opt.exp_dir+'/train.log')
        else:
            logger.warn("opt.log_file is not set")

    if not hasattr(opt, 'tensorboard_log_dir') or opt.tensorboard_log_dir is None:
        if hasattr(opt, 'exp_dir'):
            setattr(opt, 'tensorboard_log_dir', opt.exp_dir+'/logs/tfevents/')
        else:
            logger.warn("opt.tensorboard_log_dir is not set")
    if hasattr(opt, 'tensorboard_log_dir') and not os.path.exists(opt.tensorboard_log_dir):
        os.makedirs(opt.tensorboard_log_dir)

    if not hasattr(opt, 'wandb_log_dir') or opt.wandb_log_dir is None:
        if hasattr(opt, 'exp_dir'):
            # a `/wandb` will be appended by wandb
            setattr(opt, 'wandb_log_dir', opt.exp_dir+'/logs/')
        else:
            logger.warn("opt.wandb_log_dir is not set")
    if hasattr(opt, 'wandb_log_dir') and not os.path.exists(opt.wandb_log_dir):
        os.makedirs(opt.wandb_log_dir)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    # save training settings
    if opt.log_file:
        shutil.copy2(opt.config, opt.exp_dir)
    logger.info(vars(opt))

    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        # added by @memray for multiple datasets
        if opt.vocab and opt.vocab != 'none':
            vocab = torch.load(opt.vocab)
        elif opt.encoder_type == 'pretrained':
            vocab = None
        else:
            vocab = None

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # @memray: a temporary workaround, as well as train.py line 43
    if opt.model_type == "keyphrase":
        if opt.tgt_type in ["one2one", "multiple"]:
            if 'sep_indices' in fields:
                del fields['sep_indices']
        else:
            if 'sep_indices' not in fields:
                sep_indices = Field(
                    use_vocab=False, dtype=torch.long,
                    postprocessing=make_tgt, sequential=False)
                fields["sep_indices"] = sep_indices
        if 'src_ex_vocab' not in fields:
            src_ex_vocab = RawField()
            fields["src_ex_vocab"] = src_ex_vocab

    tokenizer = None
    if opt.pretrained_tokenizer:
        tokenizer = load_pretrained_tokenizer(opt.pretrained_tokenizer_name, opt.cache_dir, opt.special_vocab_path,
                                              bpe_vocab=opt.vocab, bpe_merges=opt.bpe_merges, bpe_dropout=opt.bpe_dropout)
        setattr(opt, 'vocab_size', len(tokenizer))
    if opt.data_type == 'news':
        fields = reload_news_fields(fields, opt, tokenizer)

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            # added by @memray, for loading multiple datasets
            if opt.multi_dataset:
                shard_base = "train"
                train_iter = build_dataset_iter(shard_base, fields, opt, tokenizer=tokenizer)
            else:
                train_shards = []
                for train_id in opt.data_ids:
                    shard_base = "train_" + train_id
                    train_shards.append(shard_base)
                train_iter = build_dataset_iter_multiple(train_shards, fields, opt, tokenizer=tokenizer)
        else:
            shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)

    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    if opt.valid:
        valid_iter = build_dataset_iter(
            "valid", fields, opt, is_train=False)
    else:
        valid_iter = None

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()

