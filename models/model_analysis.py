import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tools.plotting import scatter_matrix
from scipy.stats import gaussian_kde

from real_estate.data_processing.data_analysis import DataAnalysis


class ModelAnalysis():
    def run_model_analysis(
        data_file_path, xy_class, model_class,
        scatter_lims, error_density_lims
    ):
        data = ModelAnalysis.read_data(data_file_path)
        xy = xy_class(data, exclude_suburb=False)

        ModelAnalysis.model_analysis(
            data, xy,
            model_class, scatter_lims, error_density_lims
        )

    def read_data(data_file_path):
        return pd.read_hdf(data_file_path)

    def model_analysis(data, xy, model_class, scatter_lims, error_density_lims):
        filtered_data = xy.filter_data(data)
        results, model = ModelAnalysis.describe_model_estimations(
            xy, model_class)
        ModelAnalysis.model_results_analysis(filtered_data, results, xy)
        ModelAnalysis.display_model_coefs(model, xy)

        ModelAnalysis.model_graphics(filtered_data, xy)
        ModelAnalysis.model_accuracy(
            results, scatter_lims, error_density_lims)

    def describe_model_estimations(xy, model_class):
        model = model_class(xy.X.values, xy.y.values)
        estimates = model.predict()
        results = pd.DataFrame({
            'actuals': xy.y,
            'estimates': estimates,
            'error': estimates - xy.y
        })

        DataAnalysis.display_df_as_html(results.describe())
        return results, model

    def model_results_analysis(filtered_data, results, xy):
        extended_results = pd.concat(
            [filtered_data[[a for a, _ in xy.X_DATA]], results],
            axis=1, ignore_index=False
        )

        DataAnalysis.display_df_as_html(extended_results[:19])
        print('Mean absolute error: %i' % results['error'].abs().mean())

    def display_model_coefs(model, xy):
        coefs = pd.DataFrame({
            'Coef': pd.Index(['intercept'] + list(xy.X.columns.values)),
            'value': [model.model.intercept_] + list(model.model.coef_)
        })
        DataAnalysis.display_df_as_html(coefs)

    def model_graphics(filtered_data, xy):
        sns.violinplot(
            x="property_type", y="price_max",
            data=filtered_data,
            inner=None, figsize=(20, 20)
        )

        # filtered_sales.plot.density('price_max')
        # filtered_sales.plot.scatter(x='property_type', y='price_max', style='.')

        cols = list('property_type_' + filtered_data['property_type'].unique())

        # The 'Not Specified' property_type was dropped so that we have
        # identifiable coefficients.
        cols.remove('property_type_Not Specified')

        x = xy.X[cols].copy()
        x['price'] = xy.y
        scatter_matrix(x, alpha=0.2, figsize=(20, 20), diagonal='kde')

    def model_accuracy(results, scatter_lims, error_density_lims):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.scatter(results['actuals'], results['estimates'])
        max_point = results[['actuals', 'estimates']].max().max()
        ax1.plot([0, max_point], [0, max_point])
        ax1.set_xlim(scatter_lims)
        ax1.set_ylim(scatter_lims)

        density = gaussian_kde(results['error'][results['estimates']>0])
        x = np.linspace(error_density_lims[0], error_density_lims[1], 10000)
        ax2.plot(x, density(x))

        plt.show()
