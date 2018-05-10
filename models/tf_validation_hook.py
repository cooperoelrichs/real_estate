import tensorflow as tf


class ValidationHook(tf.train.SessionRunHook):
    def __init__(self, model_fn, params, batch_size, input_fn, checkpoint_dir,
                 every_n_secs=None, every_n_steps=None):
        self._iter_count = 0
        # self._estimator = tf.contrib.tpu.TPUEstimator(
        #     model_fn=model_fn,
        #     use_tpu=False,
        #     train_batch_size=batch_size,
        #     eval_batch_size=batch_size,
        #     # params={"data_dir": FLAGS.data_dir},
        #     model_dir=checkpoint_dir,
        #     # config=run_config
        # )

        self._eval_estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params=params,
            model_dir=checkpoint_dir
        )
        self._input_fn = input_fn
        self._timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
        self._should_trigger = False

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            print('Running an evaluation epoch.')
            self._eval_estimator.evaluate(
                self._input_fn
            )
            self._timer.update_last_triggered_step(self._iter_count)
            print('Evaluation complete.')
        self._iter_count += 1
