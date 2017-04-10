import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

from real_estate.data_processing.data_analysis import DataAnalysis


class DataFeaturesAnalysis():
        def run(data_file_path, outputs_dir):
            df = DataAnalysis.read_data(data_file_path)
            feature_analysis_dir = outputs_dir + 'feature_analysis_-_'
            DataFeaturesAnalysis.tables(df, feature_analysis_dir)
            DataFeaturesAnalysis.plots(df, feature_analysis_dir)

        def tables(df, feature_analysis_dir):
            pivots = [
                ('property_type', 'bedrooms'),
                ('suburb', 'bedrooms'),
                ('suburb', 'property_type'),
            ]
            DataFeaturesAnalysis.table_pivots(
                pivots, df, feature_analysis_dir
            )

            DataFeaturesAnalysis.table_counts(
                ['suburb'], df, feature_analysis_dir
            )

        def table_pivots(pivot_pairs, df, outputs_dir):
            for a, b in pivot_pairs:
                count_matrix = pd.pivot_table(
                    df.loc[:, (a, b)],
                    index=a, columns=b, aggfunc=len, fill_value=0
                )
                name = DataFeaturesAnalysis.join_names((a, b))
                file_path = outputs_dir + name + '.html'
                DataAnalysis.save_df_as_html(count_matrix, file_path)

        def table_counts(columns, df, outputs_dir):
            for column in columns:
                count = pd.DataFrame(
                    df[column].value_counts(dropna=True),
                    columns=[column]
                )
                file_path = outputs_dir + column + '_count' + '.html'
                DataAnalysis.save_df_as_html(count, file_path)

        def plots(df, feature_analysis_dir):
            bar_feature_groups = [
                (('bedrooms',), None),
                (('bathrooms',), None),
                (('garage_spaces',), None),
                (('property_type',), None)
            ]
            DataFeaturesAnalysis.plot_feature_groups(
                bar_feature_groups, df, feature_analysis_dir, 'bar'
            )

            barh_feature_groups = [
                (('suburb',), (4, 16)),
                (('property_type', 'bedrooms'), None),
                (('suburb', 'bedrooms'), (6, 16 * 4))
            ]
            DataFeaturesAnalysis.plot_feature_groups(
                barh_feature_groups, df, feature_analysis_dir, 'barh'
            )

        def plot_feature_groups(groups, df, outputs_dir, plot_type):
            for features, figure_size in groups:
                group_name = DataFeaturesAnalysis.join_names(features)
                file_name = 'property_count_by_%s.png' % group_name
                DataFeaturesAnalysis.plot_property_count_by(
                    features, df,
                    outputs_dir + file_name,
                    plot_type,
                    figure_size
                )

        def join_names(strs):
            return reduce(lambda a, b: a + '_' + b, strs)

        def plot_property_count_by(features, df, output_file, plot_type, figure_size):
            plt.figure()
            df.groupby(features).size().sort_index().plot(
                kind=plot_type, figsize=figure_size
            )
            plt.tight_layout()
            plt.savefig(output_file)
