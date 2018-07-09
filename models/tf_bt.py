import os
import sys

import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from real_estate.models.tf_model_base import TFModelBase
from real_estate.models.price_model import PriceModel


class TFBTModel(TFModelBase):
    """
    https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator
    /BoostedTreesRegressor
    """

    USE_TPU = False
    USE_GPU = True

    def __init__(
        self, name,
        n_batches_per_layer, n_trees, max_depth, learning_rate,
        l1_regularization, l2_regularization,
        tree_complexity, min_node_weight, feature_column_specs,
        batch_size, steps_between_evaluations,
        validation_split, outputs_dir, bucket_dir
    ):
        self.name = name
        self.n_batches_per_layer = n_batches_per_layer
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.tree_complexity = tree_complexity
        self.min_node_weight = min_node_weight
        self.feature_column_specs = feature_column_specs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = self.n_batches_per_layer * self.max_depth
        self.steps_between_evaluations = steps_between_evaluations

        self.model_dir = self.get_model_dir(outputs_dir, bucket_dir)
        self.outputs_dir = outputs_dir

        super().__init__(outputs_dir, bucket_dir)

    def model_checks(self):
        pass

    def compile_model(self):
        return tf.estimator.BoostedTreesRegressor(
            self.gen_feature_columns(),
            n_batches_per_layer=self.n_batches_per_layer,
            model_dir=self.model_dir,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l1_regularization=self.l1_regularization,
            l2_regularization=self.l2_regularization,
            tree_complexity=self.tree_complexity,
            min_node_weight=self.min_node_weight,
            config=self.get_simple_run_config()
        )

    def gen_feature_columns(self):
        return list(map(self.make_feature_column, self.feature_column_specs))

    def make_feature_column(self, spec):
        name, type, _, boundaries = spec
        if type == 'numeric':
            column =  tf.feature_column.numeric_column(name, dtype=tf.float32)
        elif type == 'categorical':
            # column = tf.feature_column.categorical_column_with_identity(name, 50)
            column =  tf.feature_column.numeric_column(name, dtype=tf.float32)

        return tf.feature_column.bucketized_column(
            column, self.make_boundaries(*boundaries)
        )

    def make_boundaries(self, min_, max_, n):
        return list(np.linspace(min_, max_, n))

    def save_tf_dataset(self, X, y, run_dir, mode):
        data_files = []
        x_fn = os.path.join(run_dir, 'data-x-{}.tfrecords'.format(mode))
        data_files.append(x_fn)
        with tf.python_io.TFRecordWriter(x_fn) as writer:
            for i in range(X.shape[0]):
                features = {}
                for name, _, j, _ in self.feature_column_specs:
                    features[name] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[X[i, j]])
                    )
                writer.write(tf.train.Example(
                    features=tf.train.Features(feature=features)
                ).SerializeToString())


        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            y_fn = os.path.join(run_dir, 'data-y-{}.tfrecords'.format(mode))
            data_files.append(y_fn)
            with tf.python_io.TFRecordWriter(y_fn) as writer:
                for i in range(y.shape[0]):
                    labels = {'label': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[y[i]])
                    )}

                    writer.write(tf.train.Example(
                        features=tf.train.Features(feature=labels)
                    ).SerializeToString())

        return data_files

    def make_input_fn(self, ds_files, epochs, mode):
        X_WIDTH = 5
        for file in ds_files:
            assert tf.gfile.Exists(file)

        def decode_x(example):
            features = {}
            for name, _, _, _ in self.feature_column_specs:
                features[name] = tf.FixedLenSequenceFeature(
                    shape=(1,), dtype=tf.float32, allow_missing=True
                )

            parsed_features = tf.parse_single_example(example, features)
            return parsed_features

        def decode_y(example):
            features = {
                'label': tf.FixedLenSequenceFeature(
                    shape=(1,), dtype=tf.float32, allow_missing=True
                )
            }
            parsed_features = tf.parse_single_example(example, features)
            return parsed_features['label'][0, 0]

        def input_fn():
            batch_size = self.batch_size

            if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
                shuffle = batch_size * 1000
                shuffle_and_repeat = tf.contrib.data.shuffle_and_repeat
                # batch_and_drop = tf.contrib.data.batch_and_drop_remainder

                ds_x = tf.data.TFRecordDataset(ds_files[0])
                ds_x = ds_x.map(decode_x, num_parallel_calls=8)
                ds_y = tf.data.TFRecordDataset(ds_files[1])
                ds_y = ds_y.map(decode_y, num_parallel_calls=8)

                ds = tf.data.Dataset.zip((ds_x, ds_y))
                ds = ds.cache()
                ds = ds.apply(shuffle_and_repeat(shuffle, epochs))
                # ds = ds.apply(batch_and_drop(batch_size))
            elif mode == tf.estimator.ModeKeys.PREDICT:
                ds = tf.data.TFRecordDataset(ds_files[0])
                ds = ds.map(decode_x, num_parallel_calls=8)
                ds = ds.map(decode_x, num_parallel_calls=8)
                ds = ds.cache()
                # ds = ds.batch(batch_size)


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


class TFBT(PriceModel):
    MODEL_CLASS = TFBTModel
    PARAMS = {
        'name': 'tf_bt',
        'n_batches_per_layer': 100,
        'n_trees': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'l1_regularization': 0.0,
        'l2_regularization': 0.0,
        'tree_complexity': 0.0,
        'min_node_weight': 0.0,
        'batch_size': 2**12,
        'validation_split': 0.3,
        'steps_between_evaluations': 5000,
        'outputs_dir': None,
        'bucket_dir': None,
        'feature_column_specs': [
            ('bedrooms', 'numeric', 0, (0, 10, 10)),
            ('garage_spaces', 'numeric', 1, (0, 9, 10)),
            ('bathrooms', 'numeric', 2, (0, 9, 10)),
            ('X', 'numeric', 3, (-50, 160, 1e6)),
            ('Y', 'numeric', 4, (-50, 160, 1e6)),
            ('property_type', 'categorical', 5, (0, 19, 20)),
        ]
    }
