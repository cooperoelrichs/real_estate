import os
import contextlib
import csv

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tools.plotting import scatter_matrix
from scipy.stats import gaussian_kde

from real_estate.data_processing.data_analysis import DataAnalysis
from real_estate.data_processing.data_storer import DataStorer
from real_estate.models.xy import XY
from real_estate.models.model_spec_optimisation_plotter import (
    ModelSpecOptimisationPlotter)


class ModelAnalysis():
    MAXIMUM_NUMBER_OF_RESULTS_TO_SAVE = 10**3

    def run(
        data_file_path, file_type,
        xy_class, model_class,
        scatter_lims, error_density_lims,
        outputs_dir
    ):
        data = DataStorer.read_ft(file_type, data_file_path)
        xy = ModelAnalysis.make_xy(data, xy_class)

        ModelAnalysis.model_analysis(
            data, xy,
            model_class, scatter_lims, error_density_lims,
            outputs_dir
        )

    def make_xy(data, xy_class):
        return xy_class(
            data, xy_class.GENERIC_X_SPEC, exclude_suburb=False,
            filter_on_suburb_population=True
        )

    def test_a_set_of_model_params(
        data_file_path, file_type, xy_class, model_class,
        base_params, mod_names, mod_values, outputs_dir, post_fix=None,
        log=False
    ):
        print('Testing %i combinations.\n' % len(mod_values))

        data = DataStorer.read_ft(file_type, data_file_path)
        xy = ModelAnalysis.make_xy(data, xy_class)

        if log:
            log_file = ModelAnalysis.prep_logging(outputs_dir, base_params, post_fix)

        results = []
        for i, value_set in enumerate(mod_values):
            mod = list(zip(mod_names, value_set))
            params = ModelAnalysis.modify_params(base_params, mod)
            scores = ModelAnalysis.test_model_params(
                xy, model_class, params, (i+1, len(mod_values)),
                log=log, log_file=log_file
            )
            results.append((mod, scores))

        print('\nAnalysis complete.')
        ModelAnalysis.report_on_scores(results)

        if post_fix is None:
            file_name = 'parameter_tests_-_%i_combinations.png' % (
                len(mod_values))
        else:
            file_name = 'parameter_tests_-_%i_combinations_-_%s.png' % (
                len(mod_values), post_fix)

        ModelSpecOptimisationPlotter.run(
            results, mod_names,
            os.path.join(outputs_dir, file_name))
        return results

    def plot_param_test_results(results, ordered_names, output_file):
        for x in results:
            print(x)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ModelAnalysis.scatter_plt_x_on_axis(ax1, results, scatter_lims)
        plt.savefig(output_file)

    def modify_params(params, param_mods):
        params = params.copy()
        for name, value in param_mods:
            params[name] = value
        return params

    def report_on_scores(results):
        print('\nPrinting results.')
        for param_mods, scores in results:
            print(ModelAnalysis.make_scores_report_str(param_mods, scores))

    def make_scores_report_str(param_mods, scores):
        formatted = map(ModelSpecOptimisationPlotter.strify, param_mods)
        params_str = ', '.join(formatted)
        scores_str = ': %.3f (%s)' % (
            np.mean(scores),
            ', '.join(map(lambda x: '%.5f' % x, scores))
        )
        return params_str + scores_str

    def test_model_params(
        xy, model_class, params, indicies, log, log_file
    ):
        model = model_class(
            xy.X.values, xy.y.values,
            xy.X.columns.values,
            params=params
        )
        print('\nScoring combination %i of %i.' % indicies)
        scores = model.scores()
        print('Average score: %.3f' % np.mean(scores))
        if log:
            ModelAnalysis.log_scores(log_file, scores, params, indicies)
        return scores

    def prep_logging(outputs_dir, params, post_fix):
        file_dir = os.path.join(outputs_dir, 'grid_search_results_-_%s.csv' % post_fix)
        with contextlib.suppress(FileNotFoundError):
            os.remove(file_dir)

        header = ['i'] + sorted(list(params.keys())) + ['scores', 'mean']
        with open(file_dir, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        return file_dir

    def log_scores(file_dir, scores, params, indicies):
        i, _ = indicies
        keys = sorted(list(params.keys()))
        values = [params[k] for k in keys]
        row = [i] + values + [';'.join(str(x) for x in scores), np.mean(scores)]

        import csv
        with open(file_dir, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def model_analysis(
        data, xy, model_class, scatter_lims, error_density_lims,
        outputs_dir
    ):
        filtered_data = xy.filter_data(data)
        results, model, scores, mean_error = ModelAnalysis.describe_model_estimations(
            xy, model_class,
            outputs_dir + 'discription_of_model_estimations.html',
            filtered_data
        )

        extended_results = ModelAnalysis.model_results_analysis(
            filtered_data, results, xy,
            outputs_dir + 'model_cv_estimations.html'
        )

        if model.HAS_SIMPLE_COEFS:
            ModelAnalysis.save_model_coefs(
                model, xy,
                outputs_dir + 'model_non_cv_coefficients.html'
            )

        # if model.HAS_FEATURE_IMPORTANCE:
        #     ModelAnalysis.save_feature_importance(
        #         model, xy,
        #         outputs_dir + 'non_cv_feature_importance.html'
        #     )

        # ModelAnalysis.violin_plot(
        #     filtered_data, outputs_dir + 'violin_plot.png')
        # ModelAnalysis.scatter_matrix(
        #     filtered_data, xy, outputs_dir + 'scatter_matrix.png')
        ModelAnalysis.model_accuracy(
            results, scatter_lims, error_density_lims,
            outputs_dir + 'model_accuracy_scatter_plot.png'
        )
        ModelAnalysis.model_accuracy_by_feature(
            extended_results, scatter_lims, outputs_dir
        )

        print('Mean absolute cv error: %.2f' % mean_error)
        print('Average model cv score: %.3f\n%s' % (scores.mean(), str(scores)))

    def describe_model_estimations(xy, model_class, output_file, df):
        model = model_class(
            xy.X.values, xy.y.values,
            xy.X.columns.values
        )
        scores, estimates = model.cv_score_and_predict()
        print('Average cv score: %.3f' % np.mean(scores))
        mean_absolute_error = model.mean_absolute_error(estimates)
        results = pd.DataFrame({
            'actuals': xy.y,
            'cv_estimates': estimates,
            'cv_error': estimates - xy.y,
            'cv_mean_error': mean_absolute_error,
        })

        description = results.describe()
        DataAnalysis.save_df_as_html(description, output_file)
        return results, model, scores, mean_absolute_error

    def model_results_analysis(filtered_data, results, xy, output_file):
        extended_results = pd.concat(
            [filtered_data[XY.reduce_tuples(
                [a for a, _ in xy.GENERIC_X_SPEC]
            )], results],
            axis=1, ignore_index=False
        )
        extended_results = extended_results.loc[
            :, ~ extended_results.columns.duplicated('first')
        ]
        DataAnalysis.save_df_as_html(
            extended_results[:ModelAnalysis.MAXIMUM_NUMBER_OF_RESULTS_TO_SAVE],
            output_file
        )
        return extended_results

    def save_model_coefs(model, xy, output_file):
        model.fit()
        coefs = pd.DataFrame({
            'Coef': pd.Index(['intercept'] + list(xy.X.columns.values)),
            'value': [model.model.intercept_] + list(model.model.coef_)
        })
        DataAnalysis.save_df_as_html(coefs, output_file)

    def save_feature_importance(model, xy, output_file):
        model.fit()
        feature_names = xy.X.columns.values
        raw_feature_importance = model.feature_importance()

        feature_importance = pd.DataFrame({
            'feature': list(feature_names),
            'importance': raw_feature_importance
        })

        DataAnalysis.save_df_as_html(feature_importance, output_file)

    def violin_plot(filtered_data, output_file):
        plt.figure()
        plot = sns.violinplot(
            x="property_type", y="price_max",
            data=filtered_data,
            inner=None, figsize=(20, 20)
        )
        plot.get_figure().savefig(output_file)

    def scatter_matrix(filtered_data, xy, output_file):
        # filtered_sales.plot.density('price_max')
        # filtered_sales.plot.scatter(x='property_type', y='price_max', style='.')

        cols = list('property_type_' + filtered_data['property_type'].unique())

        # The 'Not Specified' property_type was dropped so that we have
        # identifiable coefficients.
        cols.remove('property_type_Not Specified')

        x = xy.X[cols].copy()
        x['price'] = xy.y
        scatter_matrix(x, alpha=0.2, figsize=(20, 20), diagonal='kde')
        plt.savefig(output_file)

    def model_accuracy(results, scatter_lims, error_density_lims, output_file):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_ylabel('price')
        ax1.set_xlabel('model accuracy scatter plot')
        ModelAnalysis.scatter_plt_x_on_axis(ax1, results, scatter_lims)

        density = gaussian_kde(results['cv_error'][results['cv_estimates']>0])
        x = np.linspace(error_density_lims[0], error_density_lims[1], 10000)
        ax2.set_xlabel('model error density')
        ax2.plot(x, density(x))

        plt.savefig(output_file)

    def model_accuracy_by_feature(
        df, scatter_lims, outputs_dir
    ):
        plots = (
            ('bedrooms', 'num', (1, 8)),
            ('garage_spaces', 'num', (1, 8)),
            ('bathrooms', 'num', (1, 8)),
            ('property_type', 'cat', 'all'),
            # ('suburb', 'cat', 'all'),
        )

        for column, t, qq in plots:
            if t == 'num':
                ModelAnalysis.scatter_plot_numerical_column(
                    df, column, qq, scatter_lims, outputs_dir)
            elif t == 'cat':
                ModelAnalysis.scatter_plot_categorical_column(
                    df, column, qq, scatter_lims, outputs_dir)

    def scatter_plot_numerical_column(df, column, qq, scatter_lims, outputs_dir):
        x_min, x_max = qq
        x_len = x_max - x_min + 1

        img_name = 'model_accuracy_-_%s_%i_to_%i+.png' % (column, x_min, x_max)
        axes_spec = [
            (i, '>=' if i == x_max else '==', i-1)
            for i in range(x_min, x_len + 1)
        ]
        ModelAnalysis.scatter_plt_series(
            df, column, axes_spec, x_len, scatter_lims, outputs_dir, img_name)

    def scatter_plot_categorical_column(df, column, qq, scatter_lims, outputs_dir):
        if qq == 'all':
            cats = df[column].unique()
        else:
            raise RuntimeError('A qq of %s is not supported.' % qq)
        x_len = len(cats)

        img_name = 'model_accuracy_-_%s_for_%s.png' % (column, qq)
        axes_spec = [(cats[i], '==', i)for i in range(x_len)]
        ModelAnalysis.scatter_plt_series(
            df, column, axes_spec, x_len, scatter_lims, outputs_dir, img_name)

    def scatter_plt_series(
        df, column, axes_spec, x_len, scatter_lims, outputs_dir, img_name
    ):
        f, axes = plt.subplots(1, x_len, figsize=(10*x_len, 10))
        axes[0].set_ylabel('price')
        for value, op, axis_i in axes_spec:
            if op == '=>':
                x = df[df[column] >= value]
            elif op == '==':
                x = df[df[column] == value]
            axes[axis_i].set_xlabel('%s == %s' % (column, str(value)))
            ModelAnalysis.scatter_plt_x_on_axis(axes[axis_i], x, scatter_lims)
        plt.savefig(os.path.join(outputs_dir, img_name))

    def scatter_plt_x_on_axis(axis, x, scatter_lims):
        axis.scatter(x['actuals'], x['cv_estimates'])
        max_point = x[['actuals', 'cv_estimates']].max().max()
        axis.plot([0, max_point], [0, max_point])
        axis.set_xlim(scatter_lims)
        axis.set_ylim(scatter_lims)
