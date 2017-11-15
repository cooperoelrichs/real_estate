import pandas as pd

from real_estate.address_factory import AddressFactory


class Parser():
    def parse_address_strings(df):
        addresses = Parser.parse_strings(df['address_text'])
        addresses_df = Parser.create_addresses_df(addresses)

        for name in addresses_df.columns.values:
            if name in df.columns.values:
                print('Note: data set already contains a %s column.' % name)
            df[name] = addresses_df[name]
        return df

    def create_addresses_df(addresses):
        data = [a.to_tuple() for a in addresses]
        column_names = addresses[0].column_names()
        df = pd.DataFrame.from_records(data, columns=column_names)
        return df

    def parse_strings(strings):
        factory = AddressFactory()
        addresses = [factory.parse_address(s) for s in strings]
        return addresses
