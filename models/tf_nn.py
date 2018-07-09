import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from real_estate.models.tf_model_base import TFModelBase, ModelBase
from real_estate.models.price_model import PriceModel


class TFNNModel(TFModelBase):
    USE_TPU = False
    USE_GPU = False

    def __init__(
        self, name, learning_rate, learning_rate_decay, momentum,
        lambda_l1, lambda_l2, max_norm, batch_normalization, dropout_fractions,
        input_dim, epochs, batch_size, validation_split,
        layers, optimiser,
        outputs_dir, bucket_dir, steps_between_evaluations
    ):
        self.name = name
        self.input_dim = input_dim
        self.layers = layers
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.max_norm = max_norm
        self.batch_normalization = batch_normalization
        self.dropout_fractions = dropout_fractions
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimiser_name = optimiser
        self.validation_split = validation_split
        self.steps_between_evaluations = steps_between_evaluations

        super().__init__(outputs_dir, bucket_dir)

    def model_checks(self):
        if (
            not (self.dropout_fractions is False) and
            not isinstance(self.dropout_fractions, float)
        ):
            raise ValueError(
                'Dropout fractions should be False or float, instead it was %s.'
                % str(type(self.dropout_fractions))
            )

    def compile_model(self):
        estimator = tf.estimator.Estimator(
            model_fn=self.build_model_fn(),
            model_dir=self.model_dir,
            config=self.get_simple_run_config(),
            params={'batch_size': self.batch_size}
        )
        return estimator

    def build_model_fn(self):
        def model_fn(features, labels, mode, config, params):
            model = self.model_tensor(features)

            if mode == tf.estimator.ModeKeys.PREDICT:
                if self.USE_TPU:
                    return tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        predictions={'predictions': model}
                    )
                else:
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions={'predictions': model}
                    )
            elif mode == tf.estimator.ModeKeys.TRAIN:
                loss = self.loss_tensor(model, labels)
                self.summary_tensors('train-summaries', model, labels, loss)
                train_op = self.build_train_op_tensor(loss)

                if self.USE_TPU:
                    return tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=loss,
                        train_op=train_op,
                        predictions={'predictions': model},
                    )
                else:
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=loss,
                        train_op=train_op,
                        predictions={'predictions': model},
                    )
            elif mode == tf.estimator.ModeKeys.EVAL:
                loss = self.loss_tensor(model, labels)
                self.summary_tensors('eval-summaries', model, labels, loss)

                mse = tf.metrics.mean_squared_error(labels, model)
                mae = tf.metrics.mean_absolute_error(labels, model)
                r2  = self.r2_metric(labels, model)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions={'predictions': model},
                    eval_metric_ops={
                        'eval-summaries/mse': mse,
                        'eval-summaries/mae': mae,
                        'eval-summaries/r2': r2
                    }
                )
            else:
                raise ValueError("Mode '%s' not supported." % mode)
        return model_fn

    def build_train_op_tensor(self, loss):
        if self.optimiser_name == 'sgd':
            learning_rate = tf.train.inverse_time_decay(
                learning_rate=self.learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=1,
                decay_rate=self.learning_rate_decay,
            )
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=self.momentum,
                use_nesterov=True
            )

        elif self.optimiser_name == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
            )

        elif self.optimiser_name == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.learning_rate,
            )

        optimizer = self.maybe_to_tpu_optimizer(optimizer)
        train_op = self.clip_and_apply_grads(optimizer, loss)
        return train_op

    def clip_and_apply_grads(self, optimizer, loss):
        clipped_grad_var_pairs = [
            (self.clip_by_value(dx, -1., 1.), x)
            for dx, x in optimizer.compute_gradients(loss)
        ]
        train_op = optimizer.apply_gradients(
            clipped_grad_var_pairs,
            global_step=tf.train.get_global_step()
        )
        return train_op

    def clip_by_value(self, tensor, min_val, max_val):
        return tf.clip_by_value(tensor, min_val, max_val)

    def maybe_to_tpu_optimizer(self, optimizer):
        if self.USE_TPU:
            return tf.contrib.tpu.CrossShardOptimizer(optimizer)
        else:
            return optimizer

    def model_tensor(self, model):
        self.model_checks()
        regularizer = tf.contrib.layers.l1_l2_regularizer
        kernel_initializer = tf.initializers.random_uniform

        for i, units in enumerate(self.layers):
            model = tf.layers.Dense(
                units=units,
                activation=None,
                kernel_initializer=kernel_initializer(),
                kernel_regularizer=regularizer(self.lambda_l1, self.lambda_l2),
                kernel_constraint=self.maybe_max_norm(),
            )(model)

            if self.batch_normalization is True:
                model = tf.layers.BatchNormalization()(model)

            model = tf.keras.layers.PReLU()(model)

            if self.dropout_fractions and i != (len(self.layers) - 1):
                model = tf.layers.Dropout(self.dropout_fractions)(model)

        model = tf.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=kernel_initializer(),
            kernel_regularizer=regularizer(self.lambda_l1, self.lambda_l2)
        )(model)

        model = model[:, 0]
        return model

    def loss_tensor(self, model, labels):
        return tf.losses.mean_squared_error(
            labels=labels,
            predictions=model
        )

    def maybe_max_norm(self):
        if self.max_norm:
            return tf.keras.constraints.MaxNorm(self.max_norm)
        else:
            return None

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

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                shuffle = batch_size * 1000
                shuffle_and_repeat = tf.contrib.data.shuffle_and_repeat
                batch_and_drop = tf.contrib.data.batch_and_drop_remainder

                ds = ds.map(decode_x_and_y, num_parallel_calls=8)
                ds = ds.cache()
                ds = ds.apply(shuffle_and_repeat(shuffle, epochs))
                ds = ds.apply(batch_and_drop(batch_size))
            elif mode == tf.estimator.ModeKeys.PREDICT:
                ds = ds.map(decode_x_only, num_parallel_calls=8)
                ds = ds.cache().batch(batch_size)


            ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            iterator = ds.make_one_shot_iterator()
            batch = iterator.get_next()

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                X_batch, y_batch = batch
                return X_batch, y_batch
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return batch
        return input_fn


class TFNN(ModelBase):
    MODEL_CLASS = TFNNModel

    PARAMS = {
        'name': 'base_tf_nn',
        'layers': (2**8,)*5,
        'learning_rate': 1e-2,
        'learning_rate_decay': 0.01,
        'momentum': 0.95,
        'lambda_l1': 1.,
        'lambda_l2': 1.,
        'max_norm': False,
        'batch_normalization': False,
        'dropout_fractions': False,
        'epochs': 350,
        'batch_size': 2**12,
        'optimiser': 'sgd',
        'validation_split': 0.3,
        'steps_between_evaluations': 5000,

        'outputs_dir': None,
        'bucket_dir': None,
    }
