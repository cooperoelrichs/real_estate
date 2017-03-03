import numpy as np
import pandas as pd
from sklearn.cross_validation import (KFold, cross_val_score)
from sklearn.linear_model import LinearRegression


class PriceModel(object):
    def __init__(self, X, y, X_labels, df, dummy_groups):
        self.model = LinearRegression(
            fit_intercept=True, normalize=True, copy_X=True, n_jobs=1
        )
        self.X = X
        self.y = y
        self.X_labels = X_labels
        self.dummy_groups = dummy_groups
        self.df = df
        self.seed = 1

    def predict(self):
        self.model.fit(self.X, self.y)
        return self.model.predict(self.X)

    def scores(self):
        scores = np.array([
            self.score(train_i, test_i)
            for train_i, test_i in KFold(n=self.X.shape[0], n_folds=10)
        ])

        return scores

    def score(self, train_i, test_i):
        id_filter = self.identifiability_filter(self.X[train_i])
        self.model.fit(self.X[:, id_filter][train_i], self.y[train_i])
        score = self.model.score(self.X[:, id_filter][test_i], self.y[test_i])

        # TESTING ===============================
        if score < -10000:
            print('>>>>>>>>>>>>>>>>>>>>>>')
            print(self.dummy_groups)

            j = 124
            X_train = self.X[train_i, :j]
            X_test = self.X[test_i, :j]

            self.model.fit(X_train, self.y[train_i])
            score = self.model.score(X_test, self.y[test_i])

            print(score)

            print('---')
            print(self.X[train_i].shape)
            print(X_train.shape)
            print(self.X[train_i][:, id_filter].shape)

            print('---')
            print(self.X[train_i, 18:].shape)
            print(self.X[train_i, 18:j].shape)
            print(self.X[train_i][:, id_filter][:, 18:].shape)

            print('---')
            print((self.X[train_i, 18:].sum(axis=1) == 0).any())
            print((self.X[train_i, 18:j].sum(axis=1) == 0).any())
            print((self.X[train_i][:, id_filter].sum(axis=1) == 0).any())

            exit()

        return score

    def identifiability_filter(self, X_subset):
        # Ensure that every dummy group excludes at least one category.
        # Ensure that no dummy columns are empty.
        dummies_filter = np.ones((X_subset.shape[1]), dtype=bool)
        # empty_column_filter = X_subset.sum(axis=0) > 0

        for name, i_first, i_last in self.dummy_groups:
            group_filter = np.zeros((X_subset.shape[1]), dtype=bool)

            # TODO: Something is wrong with the indicies, 'i_last+1'
            #       shouldn't work.
            group_filter[i_first:i_last+1] = True
            group_columns = X_subset[:, group_filter]  # & empty_column_filter]

            any_empty_rows = (group_columns.sum(axis=1) == 0).any()
            if not any_empty_rows:
                dummies_filter[i_last - 1] = False

        return dummies_filter  # & empty_column_filter

    def cv_predict(self):
        folds = KFold(
            len(self.y), n_folds=5, shuffle=True, random_state=self.seed)

        predictions = np.zeros_like(self.y)
        for train_i, test_i in folds:
            X_train, X_test = self.X[train_i], self.X[test_i]
            y_train = self.y[train_i]

            self.model.fit(X_train, y_train)
            predictions[test_i] = self.model.predict(X_test)
        return predictions
