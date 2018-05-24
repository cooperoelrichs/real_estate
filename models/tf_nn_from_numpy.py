import os
import sys

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from real_estate.models.tf_nn import TFNN, TFNNModel


tf.logging.set_verbosity(tf.logging.INFO)


class TFNNModelFromNumpy(TFNNModel):
    def save_tf_dataset(X, y, run_dir, mode):
        run_data_dir = os.path.join(run_dir, 'data-' + mode)
        if not tf.gfile.Exists(run_data_dir):
            tf.gfile.MakeDirs(run_data_dir)

        if (
            mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL
        ):
            datasets = (('X', X), ('y', np.expand_dims(y, axis=1)))
        elif mode == tf.estimator.ModeKeys.PREDICT:
            datasets = (('X', X),)

        for name, a in datasets:
            data_file_name = os.path.join(run_data_dir, name)
            print(data_file_name)
            writer = tf.python_io.TFRecordWriter(data_file_name)

            for i in range(a.shape[0]):
                float_list = tf.train.FloatList(value=a[i, :])
                feature = tf.train.Feature(float_list=float_list)
                features = tf.train.Features(feature={'values':feature})
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
            writer.close()
        return run_data_dir

    def make_input_fn(ds_dir, epochs, mode):
        print(tf.gfile.Exists(os.path.join(ds_dir, 'X')))
        print(tf.gfile.Exists(os.path.join(ds_dir, 'y')))

        def decode_X(serialized_example):
            a = tf.parse_single_example(
                serialized_example,
                features={'values': tf.FixedLenSequenceFeature(
                    (16,), tf.float32, allow_missing=True
                )}
            )
            a = a['values']
            return a

        def decode_y(serialized_example):
            a = tf.parse_single_example(
                serialized_example,
                features={'values': tf.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True
                )}
            )
            a = a['values']
            return a

        def input_fn(params):
            X = tf.data.TFRecordDataset(filenames=os.path.join(ds_dir, 'X'))
            X = X.map(decode_X)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return X
            else:
                y = tf.data.TFRecordDataset(filenames=os.path.join(ds_dir, 'y'))
                y = y.map(decode_y)
                ds = tf.data.Dataset.zip((X, y))

                batch_size = params['batch_size']
                batched_ds = ds.prefetch(buffer_size=batch_size
                ).repeat(count=epochs
                ).shuffle(buffer_size=50000
                ).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

                batch = batched_ds.make_one_shot_iterator().get_next()
                X_batch, y_batch = batch

                X_batch = tf.reshape(X_batch, (batch_size, 16))
                y_batch = tf.reshape(y_batch, (batch_size,))
                return X_batch, y_batch

        return input_fn


class TFNNFromNumpy(TFNN):
    MODEL_CLASS = TFNNModelFromNumpy
