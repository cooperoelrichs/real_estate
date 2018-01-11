import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

from real_estate.data_processing.data_analysis import DataAnalysis
from real_estate.data_processing.data_storer import DataStorer


class DataFeaturesAnalysis():
        def run(data_file_path, file_type, outputs_dir):
            data = DataStorer.read_ft(file_type, data_file_path)
            feature_analysis_dir = outputs_dir + 'feature_analysis_-_'
            DataFeaturesAnalysis.tables(data, feature_analysis_dir)
            DataFeaturesAnalysis.plots(data, feature_analysis_dir)

        def tables(df, feature_analysis_dir):
            pivots = [
                ('property_type', 'bedrooms', 8),
                ('suburb', 'bedrooms', 8),
                ('suburb', 'property_type', None),
            ]
            DataFeaturesAnalysis.table_pivots(
                pivots, df, feature_analysis_dir
            )

            DataFeaturesAnalysis.table_counts(
                ['suburb', 'property_type'], df, feature_analysis_dir
            )

        def table_pivots(pivot_pairs, df, outputs_dir):
            for a, b, cap in pivot_pairs:
                if cap is not None:
                    count_matrix = DataFeaturesAnalysis.pivot_table_with_cap(
                        a, b, cap, df)
                else:
                    count_matrix = DataFeaturesAnalysis.pivot_table(a, b, df)
                name = DataFeaturesAnalysis.join_names((a, b))
                file_path = outputs_dir + name + '.html'
                DataAnalysis.save_df_as_html(count_matrix, file_path)

        def pivot_table(a, b, df):
            return pd.pivot_table(
                df.loc[:, (a, b)],
                index=a, columns=b, aggfunc=len, fill_value=0
            )

        def pivot_table_with_cap(a, b, cap, df):
            x = df.loc[:, (a, b)].copy()
            x.loc[x[b] > cap, b] = cap
            table = DataFeaturesAnalysis.pivot_table(a, b, x)

            cols = [a for a in table.columns]
            cols[-1] = ('%i+' % cols[-1])
            table.columns = cols
            return table

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
                (('bedrooms',), None, (None,), 12),
                (('bathrooms',), None, (None,), 12),
                (('garage_spaces',), None, (None,), 12),
                (('property_type',), None, (None,), 12)
            ]
            DataFeaturesAnalysis.plot_feature_groups(
                bar_feature_groups, df, feature_analysis_dir, 'bar'
            )

            barh_feature_groups = [
                (('suburb',), (4, 30), (None,), 6),
                (('property_type', 'bedrooms'), (6, 10), (None, 8), 8),
                # (('suburb', 'bedrooms'), (4, 16*4), (None, 8), 6)
            ]
            DataFeaturesAnalysis.plot_feature_groups(
                barh_feature_groups, df, feature_analysis_dir, 'barh'
            )

        def plot_feature_groups(groups, df, outputs_dir, plot_type):
            for features, figure_size, caps, font_size in groups:
                group_name = DataFeaturesAnalysis.join_names(features)
                file_name = 'property_count_by_%s.png' % group_name
                DataFeaturesAnalysis.plot_property_count_by(
                    features, df,
                    outputs_dir + file_name,
                    plot_type,
                    figure_size,
                    caps,
                    font_size
                )

        def join_names(strs):
            return reduce(lambda a, b: a + '_' + b, strs)

        def plot_property_count_by(
            features, df, output_file, plot_type, figure_size, caps, font_size
        ):
            x = df.loc[:, features].copy()
            for i, a in enumerate(features):
                if caps[i] is not None:
                    x.loc[x[a] > caps[i], a] = caps[i]

            plt.figure()
            counts = x.groupby(features).size().sort_index()
            counts.plot(kind=plot_type, figsize=figure_size, fontsize=font_size)
            plt.tight_layout()
            plt.savefig(output_file)
