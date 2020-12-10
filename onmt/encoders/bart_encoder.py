"""
Implementation of "Attention is All You Need"
"""
import os
import random

from typing import Any, Dict, List, Optional, Tuple
import fairseq
import torch.nn as nn
import torch

from fairseq.models.bart import BARTModel
from fairseq.models.transformer import EncoderOut

from onmt.encoders.encoder import EncoderBase
import logging

class BARTEncoder(EncoderBase):
    """
    large: 24-layer, 1024-hidden, 16-heads, 355M parameters
    """
    def __init__(self, model_name, cache_dir, max_src_length, vocab_size, opt,
                 bart_model=None, prev_checkpoint=None):
        super(BARTEncoder, self).__init__()
        self.model_name = model_name
        self.opt = opt

        if bart_model is None:
            bart_dir = os.path.join(opt.cache_dir, 'bart.large')
            bart_path = os.path.join(bart_dir, 'model.pt')
            assert os.path.exists(bart_path), 'BART checkpoint is not found! %s ' % bart_path
            logging.getLogger().info('Loading BART encoder from %s' % bart_path)

            bart_model = BARTModel.from_pretrained(bart_dir, checkpoint_file='model.pt')
        else:
            bart_model = bart_model

        if prev_checkpoint:
            bart_model.model.load_state_dict(prev_checkpoint['model'], strict=True)

        self.model = bart_model.model.encoder
        self.embed_tokens = self.model.embed_tokens
        self.embed_positions = self.model.embed_positions
        self.embed_fields = self.model.embed_tokens

        # override the forward_embedding() function to support src label embedding
        self.model.forward_embedding = forward_embedding
        self.model.forward = forward_bart_encoder

        # BART default max length of position embedding is 1024 (max_source_positions and max_target_positions)
        pos_emb_len = self.embed_positions.num_embeddings
        if max_src_length > pos_emb_len:
            emb_len = max_src_length + 8
            # new pos_embedding must be longer than src_length by at least 2 (1 for heading CLS, 1 for an offset)
            # Does fairseq start position at 2? b/c it's padding_idx is 1
            new_pos_embedding = fairseq.modules.LearnedPositionalEmbedding(emb_len, self.embed_positions.embedding_dim, padding_idx=self.embed_positions.padding_idx)
            nn.init.normal_(new_pos_embedding.weight, mean=0, std=self.embed_positions.embedding_dim ** -0.5)
            nn.init.constant_(new_pos_embedding.weight[self.embed_positions.padding_idx], 0)
            new_pos_embedding.weight.data[:pos_emb_len] = self.model.embed_positions.weight.data
            self.model.embed_positions = new_pos_embedding
            self.embed_positions = new_pos_embedding
            self.model.max_source_positions = max_src_length
            logging.getLogger().info('Adjusted position size to %d, position_embed.shape=%s'
                                     % (self.embed_positions.num_embeddings, str(self.embed_positions.weight.shape)))

        # Expand token embeddings if necessary
        token_emb_len = self.embed_tokens.num_embeddings
        if vocab_size > token_emb_len:
            new_token_embedding = nn.Embedding(vocab_size, self.embed_tokens.embedding_dim, padding_idx=self.embed_tokens.padding_idx)
            nn.init.normal_(new_token_embedding.weight, mean=0, std=self.embed_tokens.embedding_dim ** -0.5)
            nn.init.constant_(new_token_embedding.weight[self.embed_tokens.padding_idx], 0)
            new_token_embedding.weight.data[:token_emb_len] = self.model.embed_tokens.weight.data
            self.model.embed_tokens = new_token_embedding
            self.embed_tokens = new_token_embedding
            # set embed_fields to be word_embeddings, to call token embeddings easily
            self.embed_fields = new_token_embedding

            logging.getLogger().info('Adjusted vocab size to %d, token_embed.shape=%s'
                                     % (self.embed_tokens.num_embeddings, str(self.embed_tokens.weight.shape)))


    @classmethod
    def from_opt(cls, opt, embeddings, **kwargs):
        """Alternate constructor."""
        return cls(
            model_name='bart',
            cache_dir=opt.cache_dir,
            max_src_length=opt.src_seq_length_trunc,
            # vocab_size should be additionally added (after reloading fields news_dataset.reload_news_fields())
            vocab_size=opt.vocab_size,
            opt=opt,
            **kwargs
        )


    def forward(self, src, src_lengths):
        """
        :returns
            last_hidden_state:
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output: Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function.
                The Linear layer weights are trained from the next sentence prediction (classification) objective during Bert pretraining.
                This output is usually not a good summary of the semantic content of the input,
                you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
        """
        # input to BART must be batch_first, src should be (batch_size, sequence_length)
        # don't know how to add token_type_ids because embedding is processed inside
        src_tokens = src[:, :, 0].permute(1, 0)
        if src.shape[2] > 1:
            src_labels = src[:, :, 1].permute(1, 0)
        else:
            src_labels = None

        #     'encoder_out', state of the last layer # T x B x C
        #     'encoder_padding_mask',  # B x T
        #     'encoder_embedding', token embeddings (w/o positional embeddings) # B x T x C
        #     'encoder_states', states of each layer if return_all_hiddens=True # List[T x B x C]
        encoder_output = self.model(self.model, src_tokens, src_lengths,
                                    return_all_hiddens=False, src_labels=src_labels)
        # return last_hidden_state and memory_bank in shape of [src_len, batch_size, hid_dim] and length as is
        last_hidden_state = encoder_output.encoder_out

        return last_hidden_state, last_hidden_state, encoder_output


def forward_embedding(self, src_tokens, src_labels, token_embedding: Optional[torch.Tensor] = None):
    '''
    See fairseq.models.transformer.py L376, forward_embedding()
    Embed tokens and positions, both shape=[batch_size, src_len] and weights in embed_tokens
    :param self: BART model object
    :param src_tokens: text tokens
    :param src_labels: feature labels
    :return:
    '''
    if token_embedding is None:
        token_embedding = self.embed_tokens(src_tokens)
    x = embed = self.embed_scale * token_embedding

    if self.embed_positions is not None:
        x = embed + self.embed_positions(src_tokens)

    if src_labels is not None:
        x += self.embed_tokens(src_labels)

    if self.layernorm_embedding:
        x = self.layernorm_embedding(x)

    x = self.dropout_module(x)

    if self.quant_noise is not None:
        x = self.quant_noise(x)

    return x, embed


def forward_bart_encoder(self,
                         src_tokens,
                         src_lengths,
                         return_all_hiddens: bool = False,
                         src_labels=None,
                         token_embeddings: Optional[torch.Tensor] = None,
                         **unused):
    """
    Args:
        src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
        src_lengths (torch.LongTensor): lengths of each source sentence of
            shape `(batch)`
        return_all_hiddens (bool, optional): also return all of the
            intermediate hidden states (default: False).

    Returns:
        namedtuple:
            - **encoder_out** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_padding_mask** (ByteTensor): the positions of
              padding elements of shape `(batch, src_len)`
            - **encoder_embedding** (Tensor): the (scaled) embedding lookup
              of shape `(batch, src_len, embed_dim)`
            - **encoder_states** (List[Tensor]): all intermediate
              hidden states of shape `(src_len, batch, embed_dim)`.
              Only populated if *return_all_hiddens* is True.
    """
    x, encoder_embedding = self.forward_embedding(self, src_tokens, src_labels, token_embeddings)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    # compute padding mask
    encoder_padding_mask = src_tokens.eq(self.padding_idx)

    encoder_states = [] if return_all_hiddens else None

    # encoder layers
    for layer in self.layers:
        x = layer(x, encoder_padding_mask)
        if return_all_hiddens:
            assert encoder_states is not None
            encoder_states.append(x)

    if self.layer_norm is not None:
        x = self.layer_norm(x)

    return EncoderOut(
        encoder_out=x,  # T x B x C
        encoder_padding_mask=encoder_padding_mask,  # B x T
        encoder_embedding=None,  # B x T x C
        encoder_states=None,  # List[T x B x C]
        # encoder_embedding=encoder_embedding,  # B x T x C
        # encoder_states=encoder_states,  # List[T x B x C]
        src_tokens=None,
        src_lengths=None,
    )
