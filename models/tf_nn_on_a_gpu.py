import os
import sys

import numpy as np
import tensorflow as tf

from real_estate.models.tf_nn import TFNN, TFNNModel


class GPUTFNNModel(TFNNModel):
    USE_GPU = True


class GPUTFNN(TFNN):
    MODEL_CLASS = GPUTFNNModel
