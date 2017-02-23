import numpy as np
import pandas as pd
from sklearn.cross_validation import (KFold, cross_val_score)
from sklearn.linear_model import LinearRegression


class PriceModel(object):
    def __init__(self, xy):
        self.model = LinearRegression(
            fit_intercept=True, normalize=True, copy_X=True, n_jobs=1
        )
        self.xy = xy
        self.seed = 1

    def predict(self):
        X = self.xy.X()
        y = self.xy.y()
        self.model.fit(X, y)
        return self.model.predict(X)

    def scores(self):
        # scores = cross_val_score(self.model, self.X, self.y, cv=5)
        scores = [
            self.score(train_i, test_i) for
            train_i, test_i in KFold(n=self.xy.data.shape[0], n_folds=10)
        ]

        print(scores)
        exit()

        return scores

    def score(self, train_i, test_i):
        X_train, X_test = self.xy.X_cv(train_i), self.xy.X_cv(test_i)
        y_train, y_test = self.xy.y_cv(train_i), self.xy.y_cv(test_i)

        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)

        # ===========================
        if score < -10000:

            p = self.model.predict(X_test)
            x = X_test[p < 0][0]

            print(self.xy.X_labels[x==1])
            print(y_test[p < 0])
            print(p[p < 0])

            print(
                pd.DataFrame({
                    'Coef': pd.Index(['intercept'] + list(self.X_labels)),
                    'value': [self.model.intercept_] + list(self.model.coef_)
                })
            )
        # ===========================

        return score

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
