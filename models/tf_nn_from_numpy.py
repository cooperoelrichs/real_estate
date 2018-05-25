import os
import sys

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from real_estate.models.tf_nn import TFNN, TFNNModel


tf.logging.set_verbosity(tf.logging.INFO)


class TFNNModelFromNumpy(TFNNModel):
    def save_tf_dataset(self, X, y, run_dir, mode):
        if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
            datas = (('X', X), ('y', y))
        elif mode == tf.estimator.ModeKeys.PREDICT:
            datas = (('X', X),)

        files = []

        for name, data in datas:
            file_path = os.path.join(run_dir, 'data-' + mode + '-%s.csv' % name)
            files.append(file_path)
            np.savetxt(file_path, data, delimiter=",")

        return tuple(files)

    def make_input_fn(self, files, epochs, mode):
        X_WIDTH = 16

        datas = []
        for file in files:
            assert tf.gfile.Exists(file)
            datas.append(np.loadtxt(file, dtype=np.float32, delimiter=','))

        def input_fn(params):
            batch_size = params['batch_size']

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                data = (datas[0], datas[1].T)
            elif mode == tf.estimator.ModeKeys.PREDICT:
                data = datas[0]

            ds = tf.data.Dataset.from_tensor_slices(data)
            ds = ds.shuffle(buffer_size=int(1e6))
            ds = ds.repeat(epochs)
            ds = ds.prefetch(buffer_size=batch_size)
            ds = ds.batch(batch_size=batch_size)
            iterator = ds.make_one_shot_iterator()
            batch = iterator.get_next()

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                X_batch, y_batch = batch
                return X_batch, y_batch
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return batch

        return input_fn


class TFNNFromNumpy(TFNN):
    MODEL_CLASS = TFNNModelFromNumpy
