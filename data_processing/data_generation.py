from functools import reduce
from itertools import product
from operator import add

import pandas as pd
import numpy as np

from real_estate.data_processing.data_analysis import DataAnalysis


def comparison_data(name, data, outputs_dir,
                    xy_class, comparisons, model_class):
    xy = xy_class(data, exclude_suburb=False)
    model = model_class(xy.X.values, xy.y.values, xy.X.columns.values)
    model.fit()
    combinations = reduce(add, [list(product(*a)) for a in comparisons])
    X_pred = generate_comparison_data(xy.X.columns, combinations)
    y_pred = model.predict(X_pred)
    predictions = pd.DataFrame(
        [(a[0][0], a[0][1], a[0][2], a[1], a[2]) for a in combinations],
        columns=(
            'bedrooms', 'bathrooms', 'garage_spaces',
            'property_type', 'suburb'
        )
    )
    predictions['estimate'] = y_pred
    DataAnalysis.save_df_as_html(
        predictions,
        outputs_dir + '%s_comparisons.html' % name
    )
    return predictions


def generate_comparison_data(column_names, combinations):
    X = np.zeros((len(combinations), column_names.shape[0]))
    for i in np.arange(len(combinations)):
        X = populate_X_for('bedrooms', X, i, combinations, column_names)
        X = populate_X_for('bathrooms', X, i, combinations, column_names)
        X = populate_X_for('garage_spaces', X, i, combinations, column_names)
        X[i, column_names == 'property_type_%s' % combinations[i][1]] = 1
        X[i, column_names == 'suburb_%s' % combinations[i][2]] = 1
    return pd.DataFrame(X, columns=column_names.values)


def populate_X_for(feature, X, i, combinations, column_names):
    a = {'bedrooms': 0, 'bathrooms': 1, 'garage_spaces': 2}
    X[i, column_names == feature] = combinations[i][0][a[feature]]
    X[i, column_names == '%s_^2' % feature] = combinations[i][0][a[feature]] ** 2
    X[i, column_names == '%s_^3' % feature] = combinations[i][0][a[feature]] ** 3
    X[i, column_names == '%s_by_%s' % (feature, combinations[i][1])] = combinations[i][0][a[feature]]
    X[i, column_names == '%s_by_%s' % (feature, combinations[i][2])] = combinations[i][0][a[feature]]
    return X
