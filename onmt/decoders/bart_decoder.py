"""
Implementation of "Attention is All You Need"
"""
import logging
import os

from torch import Tensor

from typing import Any, Dict, List, Optional, Tuple
from fairseq.models.bart import BARTModel
from fairseq.models.fairseq_encoder import EncoderOut
from onmt.decoders.decoder import DecoderBase


class BARTDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, opt, embeddings, bart_model=None, prev_checkpoint=None):
        super(BARTDecoder, self).__init__()
        self.opt = opt

        if bart_model is None:
            bart_dir = os.path.join(opt.cache_dir, 'bart.large')
            bart_path = os.path.join(bart_dir, 'model.pt')
            assert os.path.exists(bart_path), 'BART checkpoint is not found! %s ' % bart_path
            logging.getLogger().info("Loading BART decoder from %s" % bart_path)

            bart_model = BARTModel.from_pretrained(bart_dir, checkpoint_file='model.pt')
        else:
            bart_model = bart_model

        if prev_checkpoint:
            bart_model.model.load_state_dict(prev_checkpoint['model'], strict=True)

        self.model = bart_model.model.decoder
        # override the original forward function
        self.model.forward = forward_bart_decoder
        self.model.extract_features = extract_features

        self.embed_positions = self.model.embed_positions

        # if embeddings is None:
        #     raise NotImplementedError
        #     # self.embed_tokens = self.model.embed_tokens
        # else:
        #     self.model.embed_tokens = embeddings
        #     self.embed_tokens = embeddings
        #     logging.getLogger().info('Replace BART embedding with token_embed.shape=%s'
        #                              % (str(self.model.embed_tokens.weight.shape)))

        self._std_attn_idx = -1
        self._copy_attn_idx = -1

    @classmethod
    def from_opt(cls, opt, embeddings, **kwargs):
        """Alternate constructor."""
        return cls(
            opt, embeddings, **kwargs
        )

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        pass

    def forward(self, tgt, memory_bank=None,
                step=None,
                memory_lengths=None,
                encoder_output=None,
                incremental_state=None,
                **kwargs):
        """
        :param tgt: (tgt_len, batch_size, 1)
        :param memory_bank: (src_len, batch_size, dim)
        :param step:
        :param memory_lengths: (batch_size)
        :param encoder_output: encoder_out.shape=(src_len, batch_size, dim)
        :param incremental_state: attention of previous timestep, each shape=(batch_size, num_head, 1, attn_dim)
        :param kwargs:
        :return:
        """
        # make them batch-first, (tgt_len, batch_size, 1) -> (batch_size, tgt_len)
        prev_output_tokens = tgt.squeeze(2).permute(1, 0)
        # Inputs:
        #     prev_output_tokens (LongTensor): previous decoder outputs of shape `(batch, tgt_len)`, for teacher forcing
        #     encoder_out (optional): output from the encoder, used for encoder-side attention
        #     incremental_state (dict): dictionary used for storing state during :ref:`Incremental decoding`
        #     features_only (bool, optional): only return features without applying output layer (default: False).
        # Returns a tuple:
        #     - **output** decoder's output: if features_only shape=`(batch, tgt_len, hid_dim)` else shape=`(batch, tgt_len, vocab)`
        #     - a dictionary with any model-specific outputs
        #        - **attn**: if average_attn, attn=`(batch, tgt_len, src_len)`, else `(num_head, batch, tgt_len, src_len)`
        #        - **inner_states**: output of each layer
        alignment_layer = kwargs.pop('alignment_layer', None) # None means the last layer
        alignment_heads = None  # None means return all heads and process it here
        alignment_targets = kwargs.pop('alignment_targets', [])

        output, extra = forward_bart_decoder(
            self.model,
            prev_output_tokens,
            incremental_state=incremental_state,
            encoder_out=encoder_output,
            features_only=True,
            average_attn=False,
            alignment_layer = alignment_layer,
            alignment_heads = alignment_heads,
            return_all_hiddens=False,
        )

        # (batch, tgt_len, hid_dim) -> (tgt_len, batch, hid_dim)
        dec_outs = output.transpose(0, 1).contiguous()
        attn_heads = extra['attn'][0]

        # @memray as of 20201216, fairseq disables returning attentions
        if attn_heads is not None:
            attns = {"std": attn_heads[self._std_attn_idx].transpose(0, 1).contiguous()}
            attns["copy"] = attn_heads[self._copy_attn_idx].transpose(0, 1).contiguous()
            if kwargs.pop('return_all_attention', False):
                # (num_head, batch_size, tgt_len, src_len) -> (tgt_len, batch_size, num_head, src_len)
                attns["all"] = attn_heads.permute(2, 1, 0, 3).contiguous()

            for head_id, target in enumerate(alignment_targets):
                attns["alignment_%s" % target] = attn_heads[head_id].transpose(0, 1).contiguous()
        else:
            attns = None

        return dec_outs, attns


def forward_bart_decoder(
    self,
    prev_output_tokens,
    encoder_out: Optional[EncoderOut] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    features_only: bool = False,
    full_context_alignment: bool = False,
    alignment_layer: Optional[int] = None,
    alignment_heads: Optional[int] = None,
    average_attn: bool = False,
    src_lengths: Optional[Any] = None,
    return_all_hiddens: bool = False,
):
    """
    Args:
        prev_output_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_out (optional): output from the encoder, used for
            encoder-side attention
        incremental_state (dict): dictionary used for storing state during
            :ref:`Incremental decoding`
        features_only (bool, optional): only return features without
            applying output layer (default: False).
        full_context_alignment (bool, optional): don't apply
            auto-regressive mask to self-attention (default: False).

    Returns:
        tuple:
            - the decoder's output of shape `(batch, tgt_len, vocab)`
            - a dictionary with any model-specific outputs
    """
    x, extra = extract_features(
        self,
        prev_output_tokens,
        encoder_out=encoder_out,
        incremental_state=incremental_state,
        full_context_alignment=full_context_alignment,
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
        average_attn=average_attn,
    )
    if not features_only:
        x = self.output_layer(x)
    return x, extra


def extract_features(
    self,
    prev_output_tokens,
    encoder_out: Optional[EncoderOut] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    full_context_alignment: bool = False,
    alignment_layer: Optional[int] = None,
    alignment_heads: Optional[int] = None,
    average_attn=False,
):
    """
    Similar to *forward* but only return features.

    Includes several features from "Jointly Learning to Align and
    Translate with Transformer Models" (Garg et al., EMNLP 2019).

    Args:
        full_context_alignment (bool, optional): don't apply
            auto-regressive mask to self-attention (default: False).
        alignment_layer (int, optional): return mean alignment over
            heads at this layer (default: last layer).
        alignment_heads (int, optional): only average alignment over
            this many heads (default: all heads).

    Returns:
        tuple:
            - the decoder's features of shape `(batch, tgt_len, embed_dim)`
            - a dictionary with any model-specific outputs
    """
    if alignment_layer is None:
        alignment_layer = self.num_layers - 1

    # embed positions
    positions = (
        self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state
        )
        if self.embed_positions is not None
        else None
    )

    # print(prev_output_tokens.shape)
    if incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]
        if positions is not None:
            positions = positions[:, -1:]

    # embed tokens and positions
    x = self.embed_scale * self.embed_tokens(prev_output_tokens)

    if self.quant_noise is not None:
        x = self.quant_noise(x)

    if self.project_in_dim is not None:
        x = self.project_in_dim(x)

    if positions is not None:
        x += positions

    if self.layernorm_embedding is not None:
        x = self.layernorm_embedding(x)

    x = self.dropout_module(x)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    self_attn_padding_mask: Optional[Tensor] = None
    if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

    # decoder layers
    attn: Optional[Tensor] = None
    inner_states: List[Optional[Tensor]] = [x]

    for idx, layer in enumerate(self.layers):
        # print('layer=', idx)
        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None

        x, layer_attn, _ = layer(
            x,
            encoder_out["encoder_out"][0]
            if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
            else None,
            encoder_out["encoder_padding_mask"][0]
            if (
                encoder_out is not None
                and len(encoder_out["encoder_padding_mask"]) > 0
            )
            else None,
            incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=bool((idx == alignment_layer)),
            need_head_weights=bool((idx == alignment_layer)),
        )
        inner_states.append(x)
        if layer_attn is not None and idx == alignment_layer:
            attn = layer_attn.float().to(x)

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads, [H x B x tgt_len x src_len] -> [B x tgt_len x src_len]
        if average_attn:
            attn = attn.mean(dim=0)

    if self.layer_norm is not None:
        x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
        x = self.project_out_dim(x)

    return x, {"attn": [attn], "inner_states": inner_states}