import os
import sys

import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Input, Dense, Dropout, Activation, BatchNormalization, PReLU
from tensorflow.python.layers.base import InputSpec as Input
from tensorflow.python.layers.core import Dense

from tensorflow.python.framework.errors_impl import NotFoundError

from real_estate.models.simple_nn import (
    NN, SimpleNeuralNetworkModel, EmptyKerasModel
)
from real_estate.models.price_model import PriceModel
from real_estate.models.tf_validation_hook import ValidationHook


tf.logging.set_verbosity(tf.logging.INFO)


def _signals_helper___init__(self, signals):
    self._signal_keys = []
    for key in sorted(signals.keys()):
      self._signal_keys.append(key)

def _signals_helper_as_tensor_list(signals):
    return [signals[key] for key in sorted(signals.keys())]

from tensorflow.contrib.tpu.python.tpu.tpu_estimator import _SignalsHelper
_SignalsHelper.__init__ = _signals_helper___init__
_SignalsHelper.as_tensor_list = _signals_helper_as_tensor_list


class TFNNModel(SimpleNeuralNetworkModel):
    USE_TPU = False

    def __init__(
        self, learning_rate, input_dim, epochs, batch_size, validation_split,
        outputs_dir, bucket_dir, evaluate
    ):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        # self.outputs_dir = outputs_dir
        self.model_dir = os.path.join(outputs_dir, 'model')
        self.bucket_dir = bucket_dir
        self.evaluate = evaluate

        self.del_model_dir()
        self.mk_model_dir()

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

    def del_model_dir(self):
        try:
            tf.gfile.DeleteRecursively(self.get_dir())
        except NotFoundError:
            pass

    def mk_model_dir(self):
        tf.gfile.MkDir(self.get_dir())

    def get_dir(self):
        return self.choose_dir(
            self.USE_TPU, self.model_dir, self.bucket_dir
        )

    def choose_dir(self, use_tpu, model_dir, bucket_dir):
        if use_tpu:
            return bucket_dir
        else:
            return model_dir

    def compile_model(self):
        if self.USE_TPU:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                # tpu=[os.environ['TPU_NAME']]
                tpu='c-oelrichs'
            )

            run_config = tf.contrib.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=self.get_dir(),
                session_config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True
                ),
                tpu_config=tf.contrib.tpu.TPUConfig()
            )

            estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=self.build_model_fn(),
                use_tpu=self.USE_TPU,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                predict_batch_size=self.batch_size,
                model_dir=self.get_dir(),
                config=run_config
            )
        else:
            run_config = tf.contrib.tpu.RunConfig(
                model_dir=self.get_dir(),
                session_config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True
                ),
            )
            estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=self.build_model_fn(),
                use_tpu=self.USE_TPU,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                predict_batch_size=self.batch_size,
                model_dir=self.get_dir(),
                config=run_config
            )
        return estimator

    def build_model_fn(self):
        def model_fn(features, labels, mode, config, params):
            model = self.model_tensor(features)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={'predictions': model}
                )
            elif mode == tf.estimator.ModeKeys.TRAIN:
                loss = self.loss_tensor(model, labels)
                metrics = (self.metrics_fn, (labels, model))

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                )
                if self.USE_TPU:
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
                loss = self.loss_tensor(model, labels)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions={'predictions': model},
                    eval_metric_ops={
                        'mae': tf.metrics.mean_absolute_error(labels, model),
                        'r2': self.r2_metric(labels, model)
                    }
                )
            else:
                raise ValueError("Mode '%s' not supported." % mode)
        return model_fn

    def model_tensor(self, features):
        model = Dense(units=256, activation=tf.nn.relu)(features)
        model = Dense(units=256, activation=tf.nn.relu)(model)
        model = Dense(units=256, activation=tf.nn.relu)(model)
        model = Dense(units=1)(model)
        model = model[:, 0]
        return model

    def loss_tensor(self, model, labels):
        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=model
        )
        return loss

    def metrics_fn(self, labels, predictions):
        return {
            'mae': tf.metrics.mean_absolute_error(labels, predictions),
            'r2': self.r2_metric(labels, predictions)
        }

    def r2_metric(self, labels, predictions):
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

        train_ds_dir = self.save_train_dataset(
            X_train, y_train, self.get_dir()
        )
        eval_ds_dir = self.save_eval_dataset(
            X_valid, y_valid, self.get_dir()
        )

        self.model = self.compile_model()
        train_input_fn = self.make_train_input_fn(
            train_ds_dir, self.epochs
        )

        if self.evaluate:
            hooks = self.add_hooks_for_validation([], eval_ds_dir)
        else:
            hooks = []

        num_steps = int(
            X_train.shape[0] * (1 - self.validation_split) /
            self.batch_size * self.epochs
        )
        self.model.train(
            input_fn=train_input_fn,
            max_steps=num_steps,
            hooks=hooks
        )

    def add_hooks_for_validation(self, hooks, eval_ds):
        every_n_steps = 1000
        validation_input_fn = self.make_train_input_fn(
            eval_ds, 1
        )
        return hooks + [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=self.get_dir(),
                save_steps=every_n_steps
            ),
            ValidationHook(
                self.build_model_fn(),
                {'batch_size':self.batch_size}, None,
                validation_input_fn, self.get_dir(),
                every_n_steps=every_n_steps
            )
        ]

    def evaluate(self, X_test, y_test):
        X_scaled = self.x_scaler.transform(X_test)
        X_scaled = X_scaled.astype(np.float32)
        y_test = y_test.astype(np.float32)

        input_fn = self.make_test_input_fn(
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

        predict_ds_dir = self.save_predict_dataset(
            X_scaled, self.get_dir()
        )
        predict_input_fn = self.make_predict_input_fn(
            predict_ds_dir
        )

        y_pred = self.model.predict(predict_input_fn)
        y_pred = np.array([a['predictions'] for a in y_pred])
        return r2_score(y_test, y_pred)

    # def predict(self, X_pred):
    #     X_scaled = self.x_scaler.transform(X_pred)
    #     X_scaled = X_scaled.astype(np.float32)
    #     y_pred = self.model.predict(input_fn, yield_single_examples=False)
    #     return y_pred

    def save_train_dataset(self, X, y, run_dir):
        return self.save_tf_dataset(
            X, y, run_dir, tf.estimator.ModeKeys.TRAIN
        )

    def save_eval_dataset(self, X, y, run_dir):
        return self.save_tf_dataset(
            X, y, run_dir, tf.estimator.ModeKeys.EVAL
        )

    def save_predict_dataset(self, X, run_dir):
        return self.save_tf_dataset(
            X, None, run_dir, tf.estimator.ModeKeys.PREDICT
        )

    def save_tf_dataset(self, X, y, run_dir, mode):
        data_file_path = os.path.join(run_dir, 'data-' + mode + '.tfrecords')
        print(data_file_path)

        with tf.python_io.TFRecordWriter(data_file_path) as writer:
            for i in range(X.shape[0]):
                if (mode == tf.estimator.ModeKeys.TRAIN or
                    mode == tf.estimator.ModeKeys.EVAL):
                    feature={'X': tf.train.Feature(
                             float_list=tf.train.FloatList(value=X[i])),
                             'y': tf.train.Feature(
                             float_list=tf.train.FloatList(value=[y[i]]))}
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    feature={'X': tf.train.Feature(
                             float_list=tf.train.FloatList(value=X[i]))}

                writer.write(tf.train.Example(
                    features=tf.train.Features(feature=feature)
                ).SerializeToString())

        return data_file_path

    def make_train_input_fn(self, ds_dir, epochs):
        return self.make_input_fn(
            ds_dir, epochs, tf.estimator.ModeKeys.TRAIN
        )

    def make_predict_input_fn(self, ds_dir):
        return self.make_input_fn(
            ds_dir, 1, tf.estimator.ModeKeys.PREDICT
        )

    def make_input_fn(self, data_file_path, epochs, mode):
        X_WIDTH = 16
        assert tf.gfile.Exists(data_file_path)

        def decode_x_and_y(example):
            features = {
                'X': tf.FixedLenSequenceFeature(
                    shape=(X_WIDTH,), dtype=tf.float32, allow_missing=True),
                'y': tf.FixedLenSequenceFeature(
                    shape=(1,), dtype=tf.float32, allow_missing=True)
            }
            parsed_features = tf.parse_single_example(example, features)
            return (parsed_features['X'][0], parsed_features['y'][0, 0])

        def decode_x_only(example):
            features = {
                'X': tf.FixedLenSequenceFeature(
                    shape=(X_WIDTH,), dtype=tf.float32, allow_missing=True),
            }
            parsed_features = tf.parse_single_example(example, features)
            return parsed_features['X'][0]

        def input_fn(params):
            batch_size = params['batch_size']
            ds = tf.data.TFRecordDataset(data_file_path)
            ds = ds.shuffle(buffer_size=int(1e6))
            ds = ds.repeat(epochs)
            ds = ds.prefetch(buffer_size=batch_size)
            # ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                ds = ds.apply(tf.contrib.data.map_and_batch(
                    decode_x_and_y,
                    batch_size,
                    num_parallel_batches=1,
                    # drop_remainder=True
                ))
            elif mode == tf.estimator.ModeKeys.PREDICT:
                ds = ds.apply(tf.contrib.data.map_and_batch(
                    decode_x_only,
                    batch_size,
                    num_parallel_batches=1,
                    # drop_remainder=True
                ))

            iterator = ds.make_one_shot_iterator()
            batch = iterator.get_next()

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                X_batch, y_batch = batch
                # X_batch = tf.reshape(X_batch, (batch_size, 16))
                # y_batch = tf.reshape(y_batch, (batch_size,))
                return X_batch, y_batch
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return batch
        return input_fn

    ####
    # batch_size = 3
    # ds = ds.cache().repeat().shuffle(buffer_size=50000).apply(
    #     tf.contrib.data.batch_and_drop_remainder(batch_size)
    # )
    # batch = ds.make_one_shot_iterator().get_next()
    #
    # sess = tf.Session()
    # print(sess.run(batch))
    # exit()
    ####

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


class TFNN(NN):
    MODEL_CLASS = TFNNModel

    PARAMS = {
        'learning_rate': None,
        'epochs': None,
        'batch_size': None,
        'validation_split': 0.2,
        'outputs_dir': None,
        'bucket_dir': 'gs://real-estate-modelling-temp-bucket',
        'evaluate': True
    }

    def show_live_results(self, outputs_folder, name):
        PriceModel.show_live_results(self, outputs_folder, name)

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
