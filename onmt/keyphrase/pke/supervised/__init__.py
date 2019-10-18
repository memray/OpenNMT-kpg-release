# -*- coding: utf-8 -*-
# Python Keyphrase Extraction toolkit: unsupervised models

from __future__ import absolute_import

from onmt.keyphrase.pke.supervised.api import SupervisedLoadFile
from onmt.keyphrase.pke.supervised.feature_based.kea import Kea
from onmt.keyphrase.pke.supervised.feature_based.topiccorank import TopicCoRank
from onmt.keyphrase.pke.supervised.feature_based.wingnus import WINGNUS
from onmt.keyphrase.pke.supervised.neural_based.seq2seq import Seq2Seq
