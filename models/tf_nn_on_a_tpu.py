import os
import sys

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from real_estate.models.tf_nn import TFNN, TFNNModel


tf.logging.set_verbosity(tf.logging.INFO)


class TPUTFNNModel(TFNNModel):
    USE_TPU = True


class TPUTFNN(TFNN):
    MODEL_CLASS = TPUTFNNModel
