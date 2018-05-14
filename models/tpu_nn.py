import os
import sys
import shutil

import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Input, Dense, Dropout, Activation, BatchNormalization, PReLU
from tensorflow.python.layers.base import InputSpec as Input
from tensorflow.python.layers.core import Dense

from tensorflow.python.estimator.inputs.queues import feeding_functions

from real_estate.models.simple_nn import (
    NN, SimpleNeuralNetworkModel, EmptyKerasModel
)
from real_estate.models.price_model import PriceModel
from real_estate.models.tf_validation_hook import ValidationHook


tf.logging.set_verbosity(tf.logging.DEBUG)


class TPUNeuralNetworkModel(SimpleNeuralNetworkModel):
    USE_TPU = True

    def __init__(
        self, learning_rate, input_dim, epochs, batch_size, validation_split
    ):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        # TODO:
        # self.verbosity
        # self.learning_rate
        # self.learning_rate_decay
        # self.momentum
        # self.layers
        # self.lambda_l1
        # self.lambda_l2
        # self.dropout_fractions
        # self.max_norm
        # self.activation/
        # self.batch_normalization
        # self.kernel_initializer
        # self.loss
        # self.optimizer

    def compile_model(self):
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            # tpu=[os.environ['TPU_NAME']]
            tpu='c-oelrichs'
        )

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=self.outputs_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True
            ),
            tpu_config=tf.contrib.tpu.TPUConfig(),
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=self.build_model_fn(),
            use_tpu=TPUNeuralNetworkModel.USE_TPU,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            # params={"data_dir": FLAGS.data_dir},
            model_dir=self.outputs_dir,
            config=run_config
        )
        return estimator

    def build_model_fn(self):
        def model_fn(features, labels, mode, config, params):
            model = TPUNeuralNetworkModel.model_tensor(features)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={'predictions': model}
                )
            elif mode == tf.estimator.ModeKeys.TRAIN:
                loss = TPUNeuralNetworkModel.loss_tensor(model, labels)
                metrics = (TPUNeuralNetworkModel.metrics_fn, (labels, model))

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                )
                if TPUNeuralNetworkModel.USE_TPU:
                    optimizer = tf.contrib.tpu.CrossShardOptimizer(
                        optimizer
                    )

                train_op = optimizer.minimize(
                    loss, global_step=tf.train.get_global_step()
                )
                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    predictions={'predictions': model},
                    eval_metrics=metrics
                )
            elif mode == tf.estimator.ModeKeys.EVAL:
                loss = TPUNeuralNetworkModel.loss_tensor(model, labels)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions={'predictions': model},
                    eval_metric_ops={
                        'mae': tf.metrics.mean_absolute_error(labels, model),
                        'r2': TPUNeuralNetworkModel.r2_metric(labels, model)
                    }
                )
            else:
                raise ValueError("Mode '%s' not supported." % mode)
        return model_fn

    def model_tensor(features):
        # model = Input(tensor=features)
        model = Dense(units=512, activation=tf.nn.relu)(features)
        model = Dense(units=512, activation=tf.nn.relu)(model)
        model = Dense(units=256, activation=tf.nn.relu)(model)
        model = Dense(units=1)(model)
        model = model[:, 0]
        return model

    def loss_tensor(model, labels):
        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=model
        )
        return loss

    def metrics_fn(labels, predictions):
        return {
            'mae': tf.metrics.mean_absolute_error(labels, predictions),
            'r2': TPUNeuralNetworkModel.r2_metric(labels, predictions)
        }

    def r2_metric(labels, predictions):
        sse, update_op1 = tf.metrics.mean_squared_error(labels, predictions)
        sst, update_op2 = tf.metrics.mean_squared_error(
            labels, tf.fill(tf.shape(labels), tf.reduce_mean(labels))
        )
        r2_value = tf.subtract(1.0, tf.div(sse, sst))
        return r2_value, tf.group(update_op2, update_op1)

    def fit(self, X, y):
        self.x_scaler, X_scaled = self.new_scaler(X)
        X_scaled = X_scaled.astype(np.float32)
        y = y.astype(np.float32)

        validation_split = int(X.shape[0] * self.validation_split)
        X_train = X_scaled[validation_split:]
        X_valid = X_scaled[:validation_split]
        y_train = y[validation_split:]
        y_valid = y[:validation_split]

        self.model = self.compile_model()
        train_input_fn = TPUNeuralNetworkModel.make_train_input_fn(
            X_train, y_train, self.epochs
        )
        hooks = self.add_hooks_for_validation([], X_valid, y_valid)

        num_steps = int(
            X_train.shape[0] * (1 - self.validation_split) /
            self.batch_size * self.epochs
        )
        self.model.train(
            input_fn=train_input_fn,
            max_steps=num_steps,
            hooks=hooks
        )

    def add_hooks_for_validation(self, hooks, X_valid, y_valid):
        every_n_steps = 1000
        validation_input_fn = TPUNeuralNetworkModel.make_train_input_fn(
            X_valid, y_valid, 1
        )
        return hooks + [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=self.outputs_dir,
                save_steps=every_n_steps
            ),
            ValidationHook(
                self.build_model_fn(),
                {'batch_size':self.batch_size}, None,
                validation_input_fn, self.outputs_dir,
                every_n_steps=every_n_steps
            )
        ]

    def evaluate(self, X_test, y_test):
        X_scaled = self.x_scaler.transform(X_test)
        X_scaled = X_scaled.astype(np.float32)
        y_test = y_test.astype(np.float32)

        input_fn = TPUNeuralNetworkModel.make_test_input_fn(
            X_scaled, self.batch_size
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

        input_fn = TPUNeuralNetworkModel.make_test_input_fn(
            X_scaled, self.batch_size
        )

        y_pred = self.model.predict(input_fn)
        y_pred = np.array([a['predictions'] for a in y_pred])
        return r2_score(y_test, y_pred)

    def predict(self, X_pred):
        X_scaled = self.x_scaler.transform(X_pred)
        X_scaled = X_scaled.astype(np.float32)
        y_pred = self.model.predict(input_fn, yield_single_examples=False)
        return y_pred

    def make_train_input_fn(
        X, y,
        epochs,
        shuffle=False,
        queue_capacity=10000,
        num_threads=1
    ):
        import collections
        def input_fn(params):
            batch_size = params['batch_size']

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
                num_epochs=epochs
            )

            batch = (
                queue.dequeue_many(batch_size)
                if epochs is None else queue.dequeue_up_to(batch_size)
            )

            if batch:
                # Remove the first `Tensor` in `batch`, which is the row number.
                batch.pop(0)

            if isinstance(X, np.ndarray):
                features = batch[0]
            else:
                features = dict(zip(feature_keys, batch[:len(feature_keys)]))

            target = batch[-1]

            print(features.shape, target.shape)
            features.set_shape((params['batch_size'], 16))
            target.set_shape((params['batch_size'],))
            print(features.shape, target.shape)

            return features, target
        return input_fn

    def make_test_input_fn(
        X,
        batch_size,
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
                num_epochs=1
            )

            batch = queue.dequeue_up_to(batch_size)
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

    def show_live_results(self, outputs_folder, name):
        checkpoints_dir = os.path.join(outputs_folder, 'checkpoints')
        try:
            shutil.rmtree(checkpoints_dir)
        except FileNotFoundError:
            pass
        self.model.outputs_dir = checkpoints_dir

    def model_summary(self):
        PriceModel.model_summary(self)
        # tf.summary.tensor_summary(
        #     'Model',
        #     self.MODEL_CLASS.loss_tensor(
        #         tf.placeholder(np.float32, shape=(
        #             self.model.batch_size, self.model.input_dim
        #         )),
        #         tf.placeholder(np.float32, shape=(self.model.batch_size,))
        #     ),
        #     summary_description=None,
        #     collections=None,
        #     summary_metadata=None,
        #     family=None,
        #     display_name=None
        # )
