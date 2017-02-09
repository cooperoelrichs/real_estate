import pandas as pd
import numpy as np

from IPython.display import display, HTML


class DataAnalysis():
    def run_data_analysis(data_file_path, xy_class):
        data = DataAnalysis.read_data(data_file_path)
        xy = xy_class(data, exclude_suburb=False)
        DataAnalysis.data_summary(data, xy)

    def read_data(data_file_path):
        return pd.read_hdf(data_file_path)

    def display_styler_as_html(styler):
        display(HTML(styler.render()))

    def display_df_as_html(df):
        DataAnalysis.display_styler_as_html(df.style)

    def display_df_as_html_with_nowrap(df):
        styler = df.style.set_table_styles(
            [{'selector': 'td', 'props': [('white-space', 'nowrap')]}]
        )
        DataAnalysis.display_styler_as_html(styler)

    def analyse_broken_sequences(filtered_data, xy):
        ordered_column_names_wo_price = [
            'state', 'suburb', 'postcode', 'road', 'house', 'house_number',
            'property_type', 'bedrooms', 'bathrooms', 'garage_spaces',
            # 'price_min', 'price_max',
            # 'sale_type', 'under_contract', 'under_application'
        ]
        sorted_data = filtered_data.sort_values(
            ordered_column_names_wo_price, axis=0)
        duplicated = sorted_data.duplicated(
            ordered_column_names_wo_price, keep=False)

        DataAnalysis.display_df_as_html_with_nowrap(
            sorted_data[duplicated][:9])

    def data_summary(data, xy):
        filtered_data = xy.filter_data(data)

        num_initial = data.shape[0]
        num_filtered = xy.filter_data(data).shape[0]
        num_seq_broken = data[data['sequence_broken']==False].shape[0]
        num_records = data.shape[0]

        print('Filtered from %i to %i records, %.2f remaining' % (
            num_initial, num_filtered, num_filtered / num_initial
        ))
        print('Records with broken sequences, %i of %i, %.2f broken.' % (
            num_seq_broken, num_records, num_seq_broken / num_records
        ))
        # print('---\nColumns with nulls:')
        # print(filtered_data.isnull().any(axis=0))

        DataAnalysis.display_df_as_html(filtered_data.describe())
        DataAnalysis.analyse_broken_sequences(filtered_data, xy)

    # def _(df):
    #     df['price_avg'] = (df['price_min'] + df['price_max']) / 2
    #     avg_prices = df[['suburb', 'price_avg']].groupby('suburb').mean()
    #     counts = df.groupby('suburb').size()
    #
    #     print(counts)
    #     print(avg_prices.sort_values('price_avg', ascending=False))
    #     print(df[df['suburb'] == "o'malley"])
    #     return avg_prices.sort_values('price_avg', ascending=False)
