"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import os
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec
from onmt.inputters.inputter import reload_news_fields, reload_keyphrase_fields
from onmt.inputters.news_dataset import load_pretrained_tokenizer

from onmt.modules import Embeddings, VecEmbedding, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from fairseq.models.bart import BARTModel


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    if opt.model_type == "vec" and for_encoder:
        return VecEmbedding(
            opt.feat_vec_size,
            emb_dim,
            position_encoding=opt.position_encoding,
            dropout=(opt.dropout[0] if type(opt.dropout) is list
                     else opt.dropout),
        )

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings, **kwargs):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" \
        or opt.model_type == "vec" or opt.model_type == "keyphrase" \
        else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings, **kwargs)


def build_decoder(opt, embeddings, **kwargs):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings, **kwargs)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    if opt.fairseq_model:
        # load a Fairseq model, such as BART
        model_opt = opt
        tokenizer = None
        if opt.pretrained_tokenizer:
            tokenizer = load_pretrained_tokenizer(opt.pretrained_tokenizer_name, opt.cache_dir, opt.special_vocab_path, bpe_vocab=opt.vocab, bpe_merges=opt.bpe_merges, bpe_dropout=opt.bpe_dropout)
            setattr(opt, 'vocab_size', len(tokenizer))
        if opt.data_type == 'news':
            fields = reload_news_fields(opt, tokenizer=tokenizer)
        elif opt.data_type == 'keyphrase':
            fields = reload_keyphrase_fields(opt, tokenizer=tokenizer)

    else:
        # load an ordinary OpenNMT model
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        vocab = checkpoint['vocab']
        if inputters.old_style_vocab(vocab):
            fields = inputters.load_old_vocab(
                vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
            )
        else:
            fields = vocab
    # @memray, to make tgt_field be aware of format of targets (multiple phrases)
    if opt.data_type == "keyphrase":
        fields["tgt"].type = opt.tgt_type

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat of OpenNMT when attention_dropout was not defined
    if not hasattr(model_opt, 'fairseq_model'):
        model_opt.__setattr__('fairseq_model', False)
    if not model_opt.fairseq_model:
        try:
            model_opt.attention_dropout
        except AttributeError:
            model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    if (model_opt.model_type == "text" \
            or model_opt.model_type == "vec" \
            or model_opt.model_type == "keyphrase")\
            and not model_opt.fairseq_model:
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    bart_model = None
    if model_opt.fairseq_model:
        bart_dir = os.path.join(model_opt.cache_dir, 'bart.large')
        bart_path = os.path.join(bart_dir, 'model.pt')
        assert os.path.exists(bart_path), 'BART checkpoint is not found! %s ' % bart_path

        bart_model = BARTModel.from_pretrained(bart_dir, checkpoint_file='model.pt')
        encoder = build_encoder(model_opt, src_emb, bart_model=bart_model, prev_checkpoint=checkpoint)
    else:
        encoder = build_encoder(model_opt, src_emb)
    # initialize encoder for non-pretrained-encoder models
    if not model_opt.fairseq_model:
        params_to_initiate = encoder.parameters()
        # only initiate decoder
        if model_opt.param_init != 0.0:
            for p in params_to_initiate:
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in params_to_initiate:
                if p.dim() > 1:
                    xavier_uniform_(p)

    # Build decoder.
    tgt_field = fields["tgt"]
    if not model_opt.fairseq_model:
        tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

        # Share the embedding matrix - preprocess with share_vocab required.
        if model_opt.share_embeddings:
            # src/tgt vocab should be the same if `-share_vocab` is specified.
            assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
                "preprocess with -share_vocab if you use share_embeddings"

            tgt_emb.word_lut.weight = src_emb.word_lut.weight
    else:
        # reuse the embedding processed in encoder (to include additional tokens)
        tgt_emb = encoder.model.embed_tokens

    if model_opt.fairseq_model:
        decoder = build_decoder(model_opt, tgt_emb, bart_model=bart_model, prev_checkpoint=checkpoint)
    else:
        decoder = build_decoder(model_opt, tgt_emb)

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if model_opt.fairseq_model:
        gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model.decoder.model.output_projection.in_features,
                      model.decoder.model.output_projection.out_features),
            Cast(torch.float32),
            gen_func
        )
        if checkpoint:
            generator[0].weight = model.decoder.model.output_projection.weight
    else:
        if not model_opt.copy_attn:
            if hasattr(model_opt, 'generator_function') and model_opt.generator_function == "sparsemax":
                gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size,
                          len(fields["tgt"].base_field.vocab)),
                Cast(torch.float32),
                gen_func
            )
            if model_opt.share_decoder_embeddings:
                generator[0].weight = decoder.embeddings.word_lut.weight
        else:
            tgt_base_field = fields["tgt"].base_field
            vocab_size = len(tgt_base_field.vocab)
            pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
            generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
    if not model_opt.fairseq_model:
        if checkpoint is not None:
            # This preserves backward-compat for models using customed layernorm
            def fix_key(s):
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                           r'\1.layer_norm\2.bias', s)
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                           r'\1.layer_norm\2.weight', s)
                return s

            checkpoint['model'] = {fix_key(k): v
                                   for k, v in checkpoint['model'].items()}
            # end of patch for backward compatibility

            model.load_state_dict(checkpoint['model'], strict=False)
            generator.load_state_dict(checkpoint['generator'], strict=False)
        else:
            if model_opt.param_init != 0.0:
                for p in model.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
                for p in generator.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            if model_opt.param_init_glorot:
                for p in model.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

            if hasattr(model.encoder, 'embeddings'):
                model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc)
            if hasattr(model.decoder, 'embeddings'):
                model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec)

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
