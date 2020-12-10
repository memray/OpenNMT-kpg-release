"""Module defining decoders."""
from onmt.decoders.bart_decoder import BARTDecoder
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder, TransformerLMDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "transformer_lm": TransformerLMDecoder,
           "bart": BARTDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", "TransformerLMDecoder", "BARTDecoder"]
