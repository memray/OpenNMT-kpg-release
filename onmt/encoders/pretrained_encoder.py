"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch
from transformers import AutoModel

from onmt.encoders.encoder import EncoderBase
import logging

class ExtClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(ExtClassifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        h = self.linear1(x).squeeze(-1)
        if mask:
            sent_scores = self.sigmoid(h) * mask.float()
        else:
            sent_scores = self.sigmoid(h)

        return sent_scores


class PretrainedEncoder(EncoderBase):
    """
    base: 12-layer, 768-hidden, 12-heads, 125M parameters
    large: 24-layer, 1024-hidden, 16-heads, 355M parameters
    """
    def __init__(self, model_name, cache_dir, src_length, vocab_size, opt):
        super(PretrainedEncoder, self).__init__()
        self.model_name = model_name
        self.opt = opt
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # if src is longer than default 512, add some randomly initialized positional embeddings
        pos_emb_len = self.model.embeddings.position_embeddings.weight.data.size(0)
        if(src_length > pos_emb_len):
            emb_len = src_length + 8
            # new pos_embedding must be longer than src_length by at least 2 (1 for heading CLS, 1 for an offset)
            new_pos_embeddings = nn.Embedding(emb_len, self.model.config.hidden_size)
            new_pos_embeddings.weight.data[:pos_emb_len] = self.model.embeddings.position_embeddings.weight.data
            new_pos_embeddings.weight.data[pos_emb_len:] = self.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(emb_len-pos_emb_len, 1)
            self.model.embeddings.position_embeddings = new_pos_embeddings

        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        self.model.resize_token_embeddings(vocab_size)
        self.vocab_size = self.model.config.vocab_size
        self.finetune = opt.finetune_encoder
        logging.getLogger().info('Adjusted vocab size to %d' % self.model.config.vocab_size)
        # set token_type_embeddings to be word_embeddings, to call token embeddings easily
        self.model.embeddings.token_type_embeddings = self.model.embeddings.word_embeddings

        if opt.ext_loss:
            assert len(opt.ext_loss_types) > 0
            assert len(opt.ext_loss_types) <= self.model.config.num_attention_heads
            self.ext_loss = opt.ext_loss
            self.lambda_ext_loss = opt.lambda_ext_loss
            self.ext_loss_types = opt.ext_loss_types
            self.ext_loss_decay_steps = opt.ext_loss_decay_steps

            logging.getLogger().info('Added %d extractive classifiers: %s' % (len(opt.ext_loss_types), str(opt.ext_loss_types)))
            dim_per_head = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
            self.dim_per_head = dim_per_head
            self.ext_classifiers = {}
            for ext_type in opt.ext_loss_types:
                ext_type_name = 'ext_'+ext_type
                self.ext_classifiers[ext_type_name] = ExtClassifier(dim_per_head)
                setattr(self, ext_type_name, self.ext_classifiers[ext_type_name])


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            model_name=opt.pretrained_encoder,
            cache_dir=opt.cache_dir,
            src_length=opt.src_seq_length_trunc,
            # vocab_size should be additionally added (after reloading fields news_dataset.reload_news_fields())
            vocab_size=opt.vocab_size,
            opt=opt
        )

    def forward(self, src, mask):
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
        # input to BERT must be batch_first, src and token_type_ids should be (batch_size, sequence_length)
        input_ids = src[:, :, 0].permute(1, 0)
        if src.shape[2] > 1:
            token_type_ids = src[:, :, 1].permute(1, 0)
        else:
            token_type_ids = None

        mask = mask.permute(1, 0)
        if(self.finetune):
            # last_hidden_scindertate: [batch_size, src_len, hid_dim], pooler_output: [batch_size, hid_dim]
            # last_hidden_state, pooler_output = self.model(input_ids)
            # last_hidden_state, pooler_output = self.model(input_ids, attention_mask=mask)
            last_hidden_state, pooler_output = self.model(input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        else:
            self.eval()
            with torch.no_grad():
                last_hidden_state, pooler_output = self.model(input_ids, attention_mask=mask, token_type_ids=token_type_ids)

        ext_logits = None
        if hasattr(self, 'ext_loss') and self.ext_loss:
            ext_logits = {}
            for cls_id, (cls_name, cls) in enumerate(self.ext_classifiers.items()):
                cls_input = last_hidden_state[:, :, self.dim_per_head * cls_id: self.dim_per_head * (cls_id + 1)]
                # input.shape=[batch_size, src_len, head_hid_dim], output.shape=[batch_size, src_len]
                logit = cls(cls_input)
                # make its shape to [length, batch_size]
                ext_logits[cls_name] = logit.permute((1, 0))

        # return last_hidden_state and memory_bank in shape of [src_len, batch_size, hid_dim] and length as is
        last_hidden_state = last_hidden_state.permute(1, 0, 2)

        return last_hidden_state, last_hidden_state, ext_logits
