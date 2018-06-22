import os
import sys

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from real_estate.models.tf_nn import TFNN, TFNNModel
from real_estate.tf_utilities import python3_compatibility_hacks


tf.logging.set_verbosity(tf.logging.INFO)


class ThisOptionUsesTheClipByValueOperationError(ValueError):
    def __init__(self, option_name):
        text = (
            'The option "{}" uses tensorflow\'s ClipByValue operation.'.format(
                option_name
            ) +
            '\nClipByValue is not supported by TPU\'s in TF 1.8.0.'
        )
        super().__init__()


class TPUTFNNModel(TFNNModel):
    USE_TPU = True
    SUPPORTED_TF_VERSION = '1.8.0'

    def model_checks(self):
        super().model_checks()

        # TF 1.8.0 TPU compatibility checks
        # TODO: remove these checks for TF 1.9.0

        if tf.__version__ != self.SUPPORTED_TF_VERSION:
            raise RuntimeError(
                'Only TF {} is supported. TF version is {}'.format(
                    self.SUPPORTED_TF_VERSION, tf.__version__))
        if self.max_norm:
            raise ThisOptionUsesTheClipByValueOperationError('max_norm')

    def clip_by_value(self, tensor, min_val, max_val):
        '''
        Avoid Tensorflow's clip_by_value, which uses the ClipByValue op,
        which isn't supported by XLA in Tensorflow verison 1.8.0.
        '''
        t_min = tf.minimum(tensor, max_val)
        t_max = tf.maximum(t_min, min_val)
        return t_max


class TPUTFNN(TFNN):
    MODEL_CLASS = TPUTFNNModel
