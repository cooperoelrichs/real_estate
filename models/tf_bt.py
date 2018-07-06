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


class TFBTModel():
    """https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/BoostedTreesRegressor
    """

    USE_TPU = False
    USE_GPU = False

    def __init__(
        self, name,
        outputs_dir, bucket_dir
    ):
        self.name = name

        if self.USE_TPU:
            model_dir = os.path.join(bucket_dir, 'model', self.name)
        else:
            model_dir = os.path.join(outputs_dir, 'model', self.name)

        self.model_dir = model_dir
        self.outputs_dir = outputs_dir

        TFNNModel.del_model_dir(self)
        TFNNModel.mk_model_dir(self)
        self.model_checks()

        self.model = self.compile_model()

    def model_checks(self):
        pass

    def compile_model(self):
        return tf.estimator.BoostedTreesRegressor(
            feature_columns,
            n_batches_per_layer,
            model_dir=None,
            weight_column=None,
            n_trees=100,
            max_depth=6,
            learning_rate=0.1,
            l1_regularization=0.0,
            l2_regularization=0.0,
            tree_complexity=0.0,
            min_node_weight=0.0,
        }


class TFBT(PriceModel):
    MODEL_CLASS = TFBTModel

    PARAMS = {
        'name': None
        'n_trees': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'l1_regularization': 0.0,
        'l2_regularization': 0.0,
        'tree_complexity': 0.0,
        'min_node_weight': 0.0,

        'outputs_dir': None,
        'bucket_dir': None,
    }
