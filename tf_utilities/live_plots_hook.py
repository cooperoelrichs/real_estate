import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class LivePlotsHook(tf.train.SessionRunHook):
    MIN_EPOCH = 1
    FIGSIZE = (15*2, 4*2)
    METRICS = (
        ('r2', 'train-summaries/r2', 'eval-summaries/r2'),
        ('mae', 'train-summaries/mae', 'eval-summaries/mae'),
        ('mse', 'train-summaries/mse', 'eval-summaries/mse'),
        ('loss', 'loss', 'loss'),
    )

    def __init__(
        self, name, outputs_dir, checkpoint_dir, every_n_steps
    ):
        every_n_secs = None
        self._name = name
        self._outputs_dir = outputs_dir
        self._training_dir = checkpoint_dir
        self._evaludation_dir = os.path.join(checkpoint_dir, 'eval')

        self._train_acc = None
        self._eval_acc = None

        self._timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
        self._iter_count = 0
        self._should_trigger = False

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

        self._train_acc = EventAccumulator(path=self._training_dir)
        self._eval_acc = EventAccumulator(path=self._evaludation_dir)

        _, self.axes = plt.subplots(
            1, len(self.METRICS),
            figsize=self.FIGSIZE, sharex=False
        )

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count
        )

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            self._train_acc.Reload()

            # eval_reloaded = False
            # if tf.gfile.Exists(self._evaludation_dir):
            self._eval_acc.Reload()
            eval_reloaded = True

            for i, metrics in enumerate(self.METRICS):
                metric_name, train_m, eval_m = metrics
                self.axes[i].clear()
                self.axes[i].set_title(metric_name)
                self.axes[i].set_xlabel('epoch')

                self.plot_values(train_m, i, self._train_acc, 'training')
                if eval_reloaded:
                    self.plot_values(eval_m, i, self._eval_acc, 'evaluation')

                self.axes[i].legend()

            plt.tight_layout()
            file_name = 'results-by-epoch-%s.png' % self._name
            plt.savefig(os.path.join(
                self._outputs_dir, file_name
            ))

            self._timer.update_last_triggered_step(self._iter_count)

        self._iter_count += 1

    def plot_values(self, metric, i, accumulator, label):
        x, y = self.extract_values(metric, accumulator)
        self.axes[i].plot(x, y, label=label)

    def extract_values(self, metric, accumulator):
        if metric in accumulator.Tags()['scalars']:
            x = [scalar.step for scalar in accumulator.Scalars(metric)]
            y = [scalar.value for scalar in accumulator.Scalars(metric)]
            return x[self.MIN_EPOCH:], y[self.MIN_EPOCH:]
        else:
            return [], []
