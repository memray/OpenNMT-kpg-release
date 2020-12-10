# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import sys
from logging.handlers import RotatingFileHandler
import os
logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET, rotate=False):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    if log_file and log_file != '':
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
