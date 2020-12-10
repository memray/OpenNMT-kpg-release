"""Module defining encoders."""
from onmt.encoders.bart_encoder import BARTEncoder
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.pretrained_encoder import PretrainedEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder,
           "pretrained": PretrainedEncoder, "bart": BARTEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc", "BARTEncoder"]
