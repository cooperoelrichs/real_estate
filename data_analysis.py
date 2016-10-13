class DataAnalysis(object):
    def analysis(df):
        df['price_avg'] = (df['price_min'] + df['price_max']) / 2
        avg_prices = df[['suburb', 'price_avg']].groupby('suburb').mean()
        counts = df.groupby('suburb').size()

        print(counts)
        print(avg_prices.sort_values('price_avg', ascending=False))
        print(df[df['suburb'] == "o'malley"])
        return avg_prices.sort_values('price_avg', ascending=False)
