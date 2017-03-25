import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tools.plotting import scatter_matrix
from scipy.stats import gaussian_kde

from real_estate.data_processing.data_analysis import DataAnalysis
from real_estate.models.xy import XY


class ModelAnalysis():
    def run_model_analysis(
        data_file_path, xy_class, model_class,
        scatter_lims, error_density_lims,
        outputs_dir
    ):
        data = ModelAnalysis.read_data(data_file_path)
        xy = xy_class(data, exclude_suburb=False)

        ModelAnalysis.model_analysis(
            data, xy,
            model_class, scatter_lims, error_density_lims,
            outputs_dir
        )

    def read_data(data_file_path):
        return pd.read_hdf(data_file_path)

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
        ModelAnalysis.model_results_analysis(
            filtered_data, results, xy,
            outputs_dir + 'model_estimations.html'
        )

        if model.HAS_SIMPLE_COEFS:
            ModelAnalysis.save_model_coefs(
                model, xy,
                outputs_dir + 'model_coefficients.html'
            )

        if model.HAS_FEATURE_IMPORTANCE:
            ModelAnalysis.save_feature_importance(
                model, xy,
                outputs_dir + 'feature_importance.html'
            )

        ModelAnalysis.violin_plot(
            filtered_data, outputs_dir + 'violin_plot.png')
        ModelAnalysis.scatter_matrix(
            filtered_data, xy, outputs_dir + 'scatter_matrix.png')
        ModelAnalysis.model_accuracy(
            results, scatter_lims, error_density_lims,
            outputs_dir + 'model_accuracy_scatter_plot.png'
        )

        print('Mean absolute error: %.2f' % mean_error)
        print('Average model score: %.2f\n%s' % (scores.mean(), str(scores)))

    def describe_model_estimations(xy, model_class, output_file, df):
        model = model_class(
            xy.X.values, xy.y.values,
            xy.X.columns.values, df,
        )
        scores = model.scores()

        model.fit()
        estimates = model.predict(model.X)
        mean_absolute_error = model.mean_absolute_error()
        results = pd.DataFrame({
            'actuals': xy.y,
            'estimates': estimates,
            'error': estimates - xy.y,
            'mean_error': mean_absolute_error,
        })

        DataAnalysis.save_df_as_html(results.describe(), output_file)
        return results, model, scores, mean_absolute_error

    def model_results_analysis(filtered_data, results, xy, output_file):
        extended_results = pd.concat(
            [filtered_data[XY.reduce_tuples([a for a, _ in xy.X_SPEC])], results],
            axis=1, ignore_index=False
        )
        extended_results = extended_results.loc[
            :, ~ extended_results.columns.duplicated('first')
        ]
        DataAnalysis.save_df_as_html(extended_results, output_file)

    def save_model_coefs(model, xy, output_file):
        coefs = pd.DataFrame({
            'Coef': pd.Index(['intercept'] + list(xy.X.columns.values)),
            'value': [model.model.intercept_] + list(model.model.coef_)
        })
        DataAnalysis.save_df_as_html(coefs, output_file)

    def save_feature_importance(model, xy, output_file):
        feature_names = xy.X.columns.values
        raw_feature_importance = model.feature_importance()

        importance_values = [
            raw_feature_importance.get('f%i' % x, np.nan)
            for x in np.arange(feature_names.shape[0])
        ]
        feature_importance = pd.DataFrame({
            'feature': list(feature_names),
            'importance': importance_values
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
        ax1.scatter(results['actuals'], results['estimates'])
        max_point = results[['actuals', 'estimates']].max().max()
        ax1.plot([0, max_point], [0, max_point])
        ax1.set_xlim(scatter_lims)
        ax1.set_ylim(scatter_lims)

        density = gaussian_kde(results['error'][results['estimates']>0])
        x = np.linspace(error_density_lims[0], error_density_lims[1], 10000)
        ax2.plot(x, density(x))

        plt.savefig(output_file)
