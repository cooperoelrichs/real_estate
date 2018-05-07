import os
import sys

import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, Nadam
from tensorflow.python.keras.regularizers import l1, l2, l1_l2
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.constraints import max_norm
from tensorflow.python.keras.layers import (
    Input, Dense, Dropout, Activation, BatchNormalization, PReLU
)

from tensorflow.python.estimator.inputs.queues import feeding_functions
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.ops.metrics_impl import mean as metrics_mean_fn
from tensorflow.python.ops.metrics_impl import _remove_squeezable_dimensions


from real_estate.models.simple_nn import NN, SimpleNeuralNetworkModel
from real_estate.models.price_model import PriceModel


class TPUNeuralNetworkModel(SimpleNeuralNetworkModel):
    USE_TPU = False

    def compile_model(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        run_config = tf.contrib.tpu.RunConfig(
            # cluster=tpu_cluster_resolver,
            # model_dir=self.outputs_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True
            ),
            tpu_config=tf.contrib.tpu.TPUConfig(),
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=TPUNeuralNetworkModel.model_fn,
            use_tpu=TPUNeuralNetworkModel.USE_TPU,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            # params={"data_dir": FLAGS.data_dir},
            config=run_config
        )
        return estimator

    def model_fn(features, labels, mode, config, params):
        '''
        Based on:
        https://github.com/tensorflow/tpu/blob/master/models/experimental
        /cifar_keras/cifar_keras.py
        '''
        model, loss = TPUNeuralNetworkModel.loss_tensor(features, labels)

        optimizer = tf.train.AdamOptimizer()
        if TPUNeuralNetworkModel.USE_TPU:
            optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions={'predictions': model},
            eval_metrics=(TPUNeuralNetworkModel.metrics_fn, (labels, model))
        )

    def metrics_fn(y_true, y_pred):
        return {
            'mae': tf.metrics.mean_absolute_error(labels=y_true, predictions=y_pred),
            # 'r2': TPUNeuralNetworkModel.r2_metric(y_true, y_pred)
        }

    def r2_metric(
        labels,
        predictions,
        weights=None,
        metrics_collections=None,
        updates_collections=None
    ):
        r2 = SimpleNeuralNetworkModel.r2(labels, predictions)
        return r2, None

    def loss_tensor(features, labels):
        model = Input(tensor=features)
        model = Dense(units=512, activation="relu")(model)
        model = Dense(units=1)(model)
        model = model[:, 0]

        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=model
        )
        return model, loss

    def fit(self, X_train, y_train):
        self.x_scaler, X_scaled = self.new_scaler(X_train)
        X_scaled = X_scaled.astype(np.float32)
        y_train = y_train.astype(np.float32)

        self.model = self.compile_model()
        input_fn = TPUNeuralNetworkModel.make_train_input_fn(
            X_scaled, y_train, self.batch_size
        )
        self.model.train(
            input_fn=input_fn,
            max_steps=10000  # self.train_steps
        )

    def evaluate(self, X_test, y_test):
        X_scaled = self.x_scaler.transform(X_test)
        X_scaled = X_scaled.astype(np.float32)
        y_test = y_test.astype(np.float32)

        input_fn = TPUNeuralNetworkModel.make_train_input_fn(
            X_scaled, y_test, self.batch_size
        )

        loss_and_metrics = self.model.evaluate(
            input_fn,
            steps=None,
            hooks=None,
            checkpoint_path=None,
            name=None
        )
        return loss_and_metrics

    def score(self, X_test, y_test):
        X_scaled = self.x_scaler.transform(X_test)
        X_scaled = X_scaled.astype(np.float32)
        y_test = y_test.astype(np.float32)

        input_fn = TPUNeuralNetworkModel.make_train_input_fn(X_scaled, y_test)

        y_pred = self.model.predict(input_fn)
        print(y_pred)
        print(list(y_pred))
        return r2_score(y_test, list(y_pred))

    def predict(self, X_pred):
        X_scaled = self.x_scaler.transform(X_pred)
        X_scaled = X_scaled.astype(np.float32)
        y_pred = self.model.predict(input_fn, yield_single_examples=False)
        return y_pred

    def make_train_input_fn(
        X, y,
        num_epochs=1,
        shuffle=False,
        queue_capacity=10000,
        num_threads=1
    ):
        import collections
        def input_fn(params):
            if 'batch_size' in params:
                batch_size = params['batch_size']
            else:
                batch_size = 256

            ordered_dict_data = ordered_dict_data = collections.OrderedDict(
                {'__direct_np_input__': X}
            )
            feature_keys = list(ordered_dict_data.keys())

            target_key = 'y'
            if target_key in ordered_dict_data:
                raise ValueError("X should not contain 'y'.")

            ordered_dict_data[target_key] = y

            queue = feeding_functions._enqueue_data(
                ordered_dict_data,
                queue_capacity,
                shuffle=shuffle,
                num_threads=num_threads,
                enqueue_size=batch_size,
                num_epochs=num_epochs
            )

            batch = (
                queue.dequeue_many(batch_size)
                if num_epochs is None else queue.dequeue_up_to(batch_size)
            )

            if batch:
                # Remove the first `Tensor` in `batch`, which is the row number.
                batch.pop(0)

            if isinstance(X, np.ndarray):
                features = batch[0]
            else:
                features = dict(zip(feature_keys, batch[:len(feature_keys)]))

            target = batch[-1]
            return features, target

        return input_fn

    def make_test_input_fn(
        X,
        batch_size=10000,
        num_epochs=1,
        shuffle=False,
        queue_capacity=10000,
        num_threads=1,
    ):
        import collections
        def input_fn(params):
            ordered_dict_data = ordered_dict_data = collections.OrderedDict(
                {'__direct_np_input__': X}
            )
            feature_keys = list(ordered_dict_data.keys())
            queue = feeding_functions._enqueue_data(
                ordered_dict_data,
                queue_capacity,
                shuffle=False,
                num_threads=num_threads,
                enqueue_size=batch_size,
                num_epochs=num_epochs
            )

            batch = (
                queue.dequeue_many(batch_size)
                if num_epochs is None else queue.dequeue_up_to(batch_size)
            )

            if batch:
                # Remove the first `Tensor` in `batch`, which is the row number.
                batch.pop(0)

            if isinstance(X, np.ndarray):
                features = batch[0]
            else:
                features = dict(zip(feature_keys, batch[:len(feature_keys)]))
            return features

        return input_fn

    # def new_scaler(self, x):
    # def empty_scaler(self, x):
    # def score(self, X_test, y_test):
    # def r2(y_true, y_pred):
    # def unscale(x, mean, scale):
    # def mae(y_true, y_pred):
    # def mse(y_true, y_pred):
    # def smooth_l1(y_true, y_pred):
    # def scaled_mae(y_scaler):
    # def scaled_mse(y_scaler):
    # def simple_lr_scheduler(learning_rate):


class TPUNN(NN):
    MODEL_CLASS = TPUNeuralNetworkModel
    # def model_summary(self):

    def show_live_results(self, outputs_folder, name):
        pass

    def model_summary(self):
        # PriceModel.model_summary(self)
        tf.summary.tensor_summary(
            'Model',
            self.MODEL_CLASS.loss_tensor(
                tf.placeholder(np.float32, shape=(self.model.batch_size, self.model.input_dim,)),
                tf.placeholder(np.float32, shape=(self.model.batch_size,))
            )[1],
            summary_description=None,
            collections=None,
            summary_metadata=None,
            family=None,
            display_name=None
        )
