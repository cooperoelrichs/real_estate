import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, Nadam
from tensorflow.python.keras.regularizers import l1, l2, l1_l2
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.constraints import max_norm
from tensorflow.python.keras.layers import (
    Dense, Dropout, Activation, BatchNormalization, PReLU)

from real_estate.models.simple_nn import NN, SimpleNeuralNetworkModel


# def main():
#     tf.logging.set_verbosity(tf.logging.INFO)
#
#     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
#         tpu_name,
#         zone=tpu_zone,
#         project=gcp_project
#     )
#
#     run_config = tf.contrib.tpu.RunConfig(
#         cluster=tpu_cluster_resolver,
#         model_dir=FLAGS.model_dir,
#         session_config=tf.ConfigProto(
#             allow_soft_placement=True, log_device_placement=True
#         ),
#         tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
#     )
#
#     estimator = tf.contrib.tpu.TPUEstimator(
#         model_fn=model_fn,
#         use_tpu=False,
#         train_batch_size=FLAGS.batch_size,
#         eval_batch_size=FLAGS.batch_size,
#         params={"data_dir": FLAGS.data_dir},
#         config=run_config)
#
#     estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
#     estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)
#
#
# if __name__ == "__main__":
#     tf.app.run()


class TPUNeuralNetworkModel(SimpleNeuralNetworkModel):
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
                model_fn=self.model_fn,
                use_tpu=False,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                # params={"data_dir": FLAGS.data_dir},
                config=run_config
            )
            return estimator

        def model_fn(features, labels, mode, config, params):
            v = layers.Input(tensor=features)
            fc1 = layers.Dense(units=512, activation="relu")(v)
            logits = layers.Dense(units=10)(fc1)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels
                )
            )

            optimizer = tf.train.AdamOptimizer()
            if FLAGS.use_tpu:
                optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tpu_estimator.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                predictions={
                    "classes": tf.argmax(input=logits, axis=1),
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }
            )

        def fit(self, X_train, y_train):
            estimator.train(
                input_fn=train_input_fn,
                max_steps= 10000 # self.train_steps
            )

        def predict(self, X_pred):
            results = estimator.evaluate(
                input_fn=eval_input_fn,
                steps=10000  # self.eval_steps
            )
            return results

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
