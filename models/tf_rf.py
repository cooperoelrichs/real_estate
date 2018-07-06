import os
import sys

import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.framework.errors_impl import NotFoundError

from real_estate.models.simple_nn import (
    NN, SimpleNeuralNetworkModel, EmptyKerasModel)
from real_estate.models.price_model import PriceModel
from real_estate.models.tf_nn import TFNNModel
from real_estate.tf_utilities.train_and_evaluate import train_and_evaluate
from real_estate.tf_utilities.validation_hook import ValidationHook
from real_estate.tf_utilities.live_plots_hook import LivePlotsHook


class TFRFModel():
    """https://www.tensorflow.org/api_docs/python/tf/contrib/tensor_forest
    """


class TFRF(PriceModel):
    MODEL_CLASS = TFRFModel
