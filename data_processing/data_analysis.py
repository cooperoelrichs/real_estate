import pandas as pd
import numpy as np
# from IPython.display import display, HTML

from real_estate.data_processing.data_storer import DataStorer


class DataAnalysis():
    def run(data_file_path, file_type, xy_class, outputs_dir):
        data = DataStorer.read_ft(file_type, data_file_path)
        xy = xy_class(
            data, xy_class.GENERIC_X_SPEC, exclude_suburb=False,
            filter_on_suburb_population=True
        )
        DataAnalysis.data_summary(data, xy, outputs_dir)
        xy.report_on_data_qc(data, outputs_dir)

    def display_df_as_html(df):
        DataAnalysis.display_styler_as_html(df.style)

    def display_styler_as_html(styler):
        display(HTML(styler.render()))

    def display_df_as_html_with_nowrap(df):
        styler = DataAnalysis.make_styler_with_nowrap(df)
        DataAnalysis.display_styler_as_html(styler)

    def save_df_as_html(df, file_path):
        DataAnalysis.save_styler_as_html(df.style, file_path, df.shape)

    def save_styler_as_html(styler, file_path, shape):
        print('Rendering a styler %s, this can be slow.' % str(shape))
        rendered = styler.render()
        with open(file_path, 'w') as f:
            f.write(rendered)

    def save_df_as_html_with_nowrap(df, file_path):
        styler = DataAnalysis.make_styler_with_nowrap(df)
        DataAnalysis.save_styler_as_html(styler, file_path, df.shape)

    def make_styler_with_nowrap(df):
        return df.style.set_table_styles(
            [{'selector': 'td', 'props': [('white-space', 'nowrap')]}]
        )

    def analyse_broken_sequences(filtered_data, xy, output_file):
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

        DataAnalysis.save_df_as_html_with_nowrap(
            sorted_data[duplicated], output_file)

    def data_summary(data, xy, outputs_dir):
        filtered_data = xy.filter_data(data)

        num_initial = data.shape[0]
        num_filtered = xy.filter_data(data).shape[0]
        num_seq_broken = data[data['sequence_broken']==False].shape[0]
        num_records = data.shape[0]

        with open(outputs_dir + 'data_notes.txt', 'w') as f:
            f.write(
                'Filtered from %i to %i records, %.2f remaining' % (
                    num_initial, num_filtered, num_filtered / num_initial)
            )
            f.write(
                'Records with broken sequences, %i of %i, %.2f broken.' % (
                    num_seq_broken, num_records, num_seq_broken / num_records)
            )

        DataAnalysis.save_df_as_html(
            filtered_data.describe(),
            outputs_dir + 'filtered_data_discription.html'
        )
        DataAnalysis.analyse_broken_sequences(
            filtered_data, xy,
            outputs_dir + 'dupicates.html'
        )

    # def _(df):
    #     df['price_avg'] = (df['price_min'] + df['price_max']) / 2
    #     avg_prices = df[['suburb', 'price_avg']].groupby('suburb').mean()
    #     counts = df.groupby('suburb').size()
    #
    #     print(counts)
    #     print(avg_prices.sort_values('price_avg', ascending=False))
    #     print(df[df['suburb'] == "o'malley"])
    #     return avg_prices.sort_values('price_avg', ascending=False)
