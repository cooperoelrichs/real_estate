import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.framework.errors_impl import NotFoundError

from real_estate.models.price_model import PriceModel
from real_estate.tf_utilities.validation_hook import ValidationHook
from real_estate.tf_utilities.live_plots_hook import LivePlotsHook


class TFModelBase(object):
    def __init__(self, outputs_dir, bucket_dir):
        self.model_dir = self.get_model_dir(outputs_dir, bucket_dir)
        self.outputs_dir = outputs_dir

        self.del_model_dir()
        self.mk_model_dir()
        self.model_checks()

    def get_model_dir(self, outputs_dir, bucket_dir):
        if self.USE_TPU:
            base = bucket_dir
        else:
            base = outputs_dir

        return os.path.join(base, 'model', self.name)

    def del_model_dir(self):
        try:
            tf.gfile.DeleteRecursively(self.model_dir)
        except NotFoundError:
            pass

    def mk_model_dir(self):
        tf.gfile.MkDir(self.model_dir)

    def get_simple_run_config(self):
        if self.USE_GPU:
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True,
            )
            return tf.estimator.RunConfig(
                model_dir=self.model_dir,
                session_config=session_config
            )
        else:
            session_config = tf.ConfigProto(
                log_device_placement=True, device_count={'GPU': 0}
            )
            return tf.estimator.RunConfig(
                model_dir=self.model_dir
            ).replace(session_config=session_config)

    def summary_tensors(self, name_space, model, labels, loss):
        with tf.name_scope(name_space):
            tf.summary.scalar('mse', self.mse_value(model, labels))
            tf.summary.scalar('mae', self.mae_value(model, labels))
            tf.summary.scalar('r2', self.r2_value(model, labels))

    def mse_value(self, model, labels):
        return tf.reduce_mean(tf.square(tf.subtract(labels, model)))

    def mae_value(self, model, labels):
        return tf.reduce_mean(tf.abs(tf.subtract(labels, model)))

    def r2_value(self, model, labels):
        sse = self.mae_value(model, labels)
        sst = self.mae_value(
            tf.fill(tf.shape(labels), tf.reduce_mean(labels)), labels
        )
        return tf.subtract(1.0, tf.div(sse, sst))

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

        train_ds_dirs = self.save_train_dataset(X_train, y_train, self.model_dir)
        eval_ds_dirs = self.save_eval_dataset(X_valid, y_valid, self.model_dir)

        self.model = self.compile_model()
        train_input_fn = self.make_train_input_fn(train_ds_dirs, self.epochs)
        eval_input_fn = self.make_train_input_fn(eval_ds_dirs, 1)

        training_steps = int(
            X_train.shape[0] * (1 - self.validation_split) /
            self.batch_size * self.epochs
        )

        evaluation_steps = int(X_train.shape[0] / self.batch_size)
        hooks = self.add_hooks_for_validation([], eval_ds_dirs)
        self.model.train(
            input_fn=train_input_fn,
            max_steps=training_steps,
            hooks=hooks
        )

    def new_scaler(self, x):
        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        return scaler, x_scaled

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
        y_test = y_test.astype(np.float32)
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def predict(self, X_pred):
        X_scaled = self.x_scaler.transform(X_pred)
        X_scaled = X_scaled.astype(np.float32)

        predict_ds_dir = self.save_predict_dataset(X_scaled, self.model_dir)
        predict_input_fn = self.make_predict_input_fn(predict_ds_dir)
        y_pred = self.model.predict(predict_input_fn)  # yield_single_examples=False
        y_pred = np.array([a['predictions'] for a in y_pred])
        return y_pred

    def add_hooks_for_validation(self, hooks, eval_ds):
        validation_input_fn = self.make_train_input_fn(eval_ds, 1)
        return hooks + [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=self.model_dir,
                save_steps=self.steps_between_evaluations
            ),
            ValidationHook(
                self.model, validation_input_fn,
                every_n_steps=self.steps_between_evaluations
            ),
            LivePlotsHook(
                self.name, self.outputs_dir, self.model_dir,
                every_n_steps=self.steps_between_evaluations
            )
        ]

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

    def make_train_input_fn(self, ds_files, epochs):
        return self.make_input_fn(
            ds_files, epochs, tf.estimator.ModeKeys.TRAIN
        )

    def make_predict_input_fn(self, ds_files):
        return self.make_input_fn(
            ds_files, 1, tf.estimator.ModeKeys.PREDICT
        )


class ModelBase(PriceModel):
    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        else:
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)
