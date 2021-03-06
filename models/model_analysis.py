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
    DATE_COLUMN = 'last_encounted'

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
        return xy_class(data, xy_class.GENERIC_X_SPEC)

    def write_xy(xy, dir, file_name):
        for df, a in ((xy.X, 'X'), (xy.y, 'y')):
            extended_file_name = ModelAnalysis.xy_f_name(file_name, a, 'csv')
            file_path = os.path.join(dir, extended_file_name)
            print('Writing XY data set to: %s' % extended_file_name)
            df.to_csv(file_path)

    def read_xy(xy_class, dir, file_name, date_columns=[]):
        X = pd.read_csv(
            os.path.join(dir, ModelAnalysis.xy_f_name(file_name, 'X', 'csv')),
            index_col=0,
        )

        y = pd.read_csv(
            os.path.join(dir, ModelAnalysis.xy_f_name(file_name, 'y', 'csv')),
            index_col=0, header=None,
            dtype=np.float64
        )

        # y is expected to be a series with shape (n,) rather than a DataFrame
        # with shape (n, 1).
        y = y[1]

        for name in date_columns:
            X[name] = pd.to_datetime(
                X[name], format='%Y-%m-%d %H:%M:%S'
            )
        return xy_class(X, y)

    def xy_f_name(base, a, file_type):
        return base + '_%s.%s' % (a, file_type)

    def test_a_set_of_model_params(
        xy, model_class,
        base_params, mod_names, mod_values, outputs_dir, n_folds,
        post_fix=None, log=False
    ):
        print('Testing %i combinations.\n' % len(mod_values))

        if log:
            log_file = ModelAnalysis.prep_logging(outputs_dir, base_params, post_fix)

        results = []
        for i, value_set in enumerate(mod_values):
            mod = list(zip(mod_names, value_set))
            mod_str = ', '.join(['-'.join((a, str(b))) for a, b in mod])
            params = ModelAnalysis.modify_params(base_params, mod)
            scores = ModelAnalysis.test_model_params(
                xy, model_class, params, (i+1, len(mod_values)), n_folds,
                log, log_file, outputs_dir, mod_str
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

        plt.clf()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ModelAnalysis.scatter_plt_actuals_v_estimates(ax1, results, scatter_lims)
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
        xy, model_class, params, indicies, n_folds,
        log, log_file,
        outputs_dir, test_name
    ):
        print('\nScoring combination %i of %i.' % indicies)
        model = model_class(
            xy.X.values, xy.y.values,
            xy.X.columns.values,
            params=params
        )

        model.model_summary()
        model.show_live_results(outputs_dir, test_name)

        scores = model.scores(n_folds)
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

        extended_results = ModelAnalysis.extend_results(
            filtered_data, results, xy
        )
        ModelAnalysis.model_results_analysis(
            extended_results, outputs_dir + 'model_cv_estimations.html'
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

    def extend_results(filtered_data, results, xy, include_date):
        extended_results = pd.concat(
            [filtered_data[XY.reduce_tuples(
                [a for a, _ in xy.GENERIC_X_SPEC]
            )], results],
            axis=1, ignore_index=False
        )

        if include_date:
            extended_results['date'] = filtered_data[ModelAnalysis.DATE_COLUMN]

        extended_results = extended_results.loc[
            :, ~ extended_results.columns.duplicated('first')
        ]
        return extended_results

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

    def model_results_analysis(extended_results, output_file):
        DataAnalysis.save_df_as_html(
            extended_results[:ModelAnalysis.MAXIMUM_NUMBER_OF_RESULTS_TO_SAVE],
            output_file
        )

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
        plt.clf()
        scatter_matrix(x, alpha=0.2, figsize=(20, 20), diagonal='kde')
        plt.savefig(output_file)

    def model_accuracy(results, scatter_lims, error_density_lims, output_file):
        plt.clf()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_ylabel('price')
        ax1.set_xlabel('model accuracy scatter plot')
        ModelAnalysis.scatter_plt_actuals_v_estimates(ax1, results, scatter_lims)

        density = gaussian_kde(results['cv_error'][results['cv_estimates']>0])
        x = np.linspace(error_density_lims[0], error_density_lims[1], 10000)
        ax2.set_xlabel('model error density')
        ax2.plot(x, density(x))

        plt.savefig(output_file)

    def plot_errors_by_time(
        df, price_range, error_range, outputs_dir
    ):
        for img_name, y in (
            ('normalised_errors_by_time.png', ModelAnalysis.normalised_error(df)),
            ('errors_by_time.png', df['cv_error'])
        ):
            plt.clf()
            f, axis = plt.subplots(1, 1, figsize=(20, 10))
            axis.set_ylabel('price')
            axis.set_xlabel('model accuracy scatter plot')

            axis.plot(df['date'], y, '.')
            # axis.set_ylim(error_range)

            plt.savefig(os.path.join(outputs_dir, img_name))

    def plot_average_daily_errors(df, outputs_dir, kind):
        ModelAnalysis.plot_averaged_values_by_group(
            df['cv_error'].abs(),
            [df['date'].dt.year, df['date'].dt.month, df['date'].dt.day],
            outputs_dir, 'average_daily_abs_errors_-_%s.png' % kind, kind
        )

    def plot_average_weekly_errors(df, outputs_dir, kind):
        ModelAnalysis.plot_averaged_values_by_group(
            df['cv_error'].abs(),
            [df['date'].dt.year, df['date'].dt.month, df['date'].dt.week],
            outputs_dir, 'average_weekly_abs_errors_-_%s.png' % kind, kind
        )

    def plot_average_monthly_errors(df, outputs_dir, kind):
        ModelAnalysis.plot_averaged_values_by_group(
            df['cv_error'].abs(),
            [df['date'].dt.year, df['date'].dt.month],
            outputs_dir, 'average_monthly_abs_errors_-_%s.png' % kind, kind
        )

    def plot_averaged_values_by_group(values, group_by, outputs_dir, img_name, kind):
        plt.clf()
        values.groupby(group_by).mean().plot(kind=kind)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, img_name))

    ACCURACY_BY_FEATURE_PLOTS = (
        ('bedrooms', 'num', (1, 8)),
        ('garage_spaces', 'num', (1, 8)),
        ('bathrooms', 'num', (1, 8)),
        ('property_type', 'cat', 'all'),
        # ('suburb', 'cat', 'all'),
    )

    def normalised_model_accuracy_by_feature(df, price_r, error_r, outputs_dir):
        df['normalised_error'] = ModelAnalysis.normalised_error(df)
        ModelAnalysis.generalised_scatter_plots_by_feature(
            df, 'actuals', 'normalised_error', (price_r, error_r), outputs_dir,
            'normalised_error', True
        )

    def normalised_error(df):
        return (df['cv_error'] / df['actuals']).abs()

    def model_accuracy_by_feature(df, price_r, outputs_dir):
        ModelAnalysis.generalised_scatter_plots_by_feature(
            df, 'actuals', 'cv_estimates', (price_r, price_r), outputs_dir,
            'model_accuracy', False
        )

    def generalised_scatter_plots_by_feature(
        df, value_1, value_2, scatter_lims, outputs_dir, img_name, normalised
    ):
        for feature_name, t, qq in ModelAnalysis.ACCURACY_BY_FEATURE_PLOTS:
            if t == 'num':
                ModelAnalysis.scatter_plot_numerical_column(
                    df[feature_name], feature_name, df[value_1], df[value_2],
                    qq, scatter_lims, normalised, outputs_dir,
                    '%s_-_%s_%i_to_%i+.png' % (img_name, feature_name, qq[0], qq[1])
                )
            elif t == 'cat':
                ModelAnalysis.scatter_plot_categorical_column(
                    df[feature_name], feature_name, df[value_1], df[value_2],
                    qq, scatter_lims, normalised, outputs_dir,
                    '%s_-_%s_for_%s.png' % (img_name, feature_name, qq)
                )

    def scatter_plot_numerical_column(
        feature, feature_name, x1, x2,
        qq, scatter_lims, normalised, outputs_dir, img_name
    ):
        x_min, x_max = qq
        x_len = x_max - x_min + 1

        axes_spec = [
            (i, '>=' if i == x_max else '==', i-1)
            for i in range(x_min, x_len + 1)
        ]
        ModelAnalysis.scatter_plt_series(
            feature, feature_name, x1, x2,
            axes_spec, x_len, scatter_lims, normalised, outputs_dir, img_name
        )

    def scatter_plot_categorical_column(
        feature, feature_name, x1, x2,
        qq, scatter_lims, normalised, outputs_dir, img_name
    ):
        if qq == 'all':
            cats = feature.unique()
        else:
            raise RuntimeError('A qq of %s is not supported.' % qq)
        x_len = len(cats)

        axes_spec = [(cats[i], '==', i)for i in range(x_len)]
        ModelAnalysis.scatter_plt_series(
            feature, feature_name, x1, x2,
            axes_spec, x_len, scatter_lims, normalised, outputs_dir, img_name
        )

    def scatter_plt_series(
        feature, feature_name, x1, x2,
        axes_spec, x_len, scatter_lims, normalised, outputs_dir, img_name
    ):
        plt.clf()
        f, axes = plt.subplots(1, x_len, figsize=(10*x_len, 10))
        axes[0].set_ylabel('price')
        for value, op, axis_i in axes_spec:
            if op == '>=':
                filter_ = feature >= value
            elif op == '==':
                filter_ = feature == value
            else:
                raise ValueError('Operation "%s" not understood.' % op)

            axes[axis_i].set_xlabel('%s == %s' % (feature_name, str(value)))
            ModelAnalysis.scatter_plt_on_axis(
                axes[axis_i], x1[filter_], x2[filter_], scatter_lims, normalised
            )

        plt.savefig(os.path.join(outputs_dir, img_name))

    def scatter_plt_actuals_v_estimates(axis, df, scatter_lims):
        ModelAnalysis.scatter_plt_on_axis(
            axis,
            x['actuals'] ,x['cv_estimates'],
            scatter_lims
        )

    def scatter_plt_on_axis(axis, x, y, scatter_lims, normalised):
        axis.scatter(x, y)
        max_value = max((x.max(), y.max()))
        if normalised:
            pass
        else:
            axis.plot([0, max_value], [0, max_value])
        axis.set_xlim(scatter_lims[0])
        axis.set_ylim(scatter_lims[1])
