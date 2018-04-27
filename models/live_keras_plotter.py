import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class LivePlotter(Callback):
    A = 'val_'
    def __init__(self, figsize, num_epochs, outputs_dir):
        super().__init__()
        self.figsize = figsize
        self.num_epochs = num_epochs
        self.outputs_dir = outputs_dir
        self.metric_names = []
        self.data = {}
        self.epochs = []

    def on_train_begin(self, _):
        for name in self.params['metrics']:
            if not name.startswith('val_'):
                self.metric_names.append(name)
                self.data[name] = {
                    't': [],
                    'v': []
                }

        _, self.axes = plt.subplots(
            1, len(self.metric_names),
            figsize=self.figsize, sharex=False
        )

    def on_epoch_end(self, epoch, results):
        self.epochs.append(epoch+1)
        for i, name in enumerate(self.metric_names):
            self.data[name]['t'].append(results[name])
            self.data[name]['v'].append(results[self.A + name])

            self.axes[i].clear()
            self.axes[i].set_title(name)
            self.axes[i].set_xlabel('epoch')
            self.axes[i].plot(self.epochs, self.data[name]['t'], label='training')
            self.axes[i].plot(self.epochs, self.data[name]['v'], label='validation')
            self.axes[i].legend()

        # plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        if (epoch + 1) == self.num_epochs:
            plt.savefig(os.path.join(self.outputs_dir, 'results-by-epoch.png'))
