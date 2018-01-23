import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class ModelSpecOptimisationPlotter():
    def run(results, ordered_names, output_file):
        plot = []
        for n in ordered_names:
            plot.append(SubPlotSpec(n))
            for params, result in results:
                sl = SubLineSpec(np.mean(result))
                for ni, v in params:
                    if n == ni:
                        sl.value = v
                    else:
                        sl.spec.append(v)
                        sl.names.append(ni)
                plot[-1].get_line(
                    sl.spec, sl.names
                ).add_result(sl.value, sl.result)

        ModelSpecOptimisationPlotter.plot(plot, output_file)

    def plot(plot, output_file):
        f, axes = plt.subplots(1, len(plot), figsize=(10*len(plot), 10))
        axes[0].set_ylabel('score')
        for i, sp in enumerate(plot):
            legend = []
            for l in sp.lines:
                axes[i].plot(l.values, l.results, '-o')
                legend.append(
                    ModelSpecOptimisationPlotter.legend_name(l.names, l.spec)
                )
            axes[i].set_xlabel(sp.name)
            axes[i].legend(legend, prop={'size': 4})
        plt.savefig(output_file, dpi=500)

    def legend_name(names, values):
        return ', '.join(map(ModelSpecOptimisationPlotter.strify, zip(names, values)))

    def strify(x):
        n, v = x
        if isinstance(v, float):
            return '%s %.3f' % (n, v)
        elif isinstance(v, int):
            return '%s %i' % (n, v)
        else:
            raise RuntimeError('What is this: %s, %s' % (str(v), str(type(v))))


class SubPlotSpec(object):
    def __init__(self, name):
        self.name = name
        self.lines = []

    def get_line(self, spec, names):
        matches = [l for l in self.lines if l.spec == spec]
        if len(matches) > 1:
            raise RuntimeError('Multiple lines with the same spec!\n%s' % str(matches))
        elif len(matches) == 0:
            self.lines.append(LineSpec(spec, names))
            return self.lines[-1]
        else:
            return matches[0]


class LineSpec(object):
    def __init__(self, spec, names):
        self.spec = spec
        self.names = names
        self.values = []
        self.results = []

    def add_result(self, v, r):
        self.values.append(v)
        self.results.append(r)


class SubLineSpec(object):
    def __init__(self, result):
        self.value = None
        self.spec = []
        self.names = []
        self.result = result
