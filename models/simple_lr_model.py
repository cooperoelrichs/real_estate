import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


class PriceModel(object):
    def __init__(self, X, y):
        self.model = LinearRegression(
            fit_intercept=True, normalize=True, copy_X=True, n_jobs=1
        )
        self.X = X
        self.y = y
        self.seed = 1

    def predict(self):
        self.model.fit(self.X, self.y)
        return self.model.predict(self.X)

    def cv_predict(self):
        folds = cross_validation.KFold(
            len(self.y), n_folds=5, shuffle=True, random_state=self.seed)

        predictions = np.zeros_like(self.y)
        for train_i, test_i in folds:
            X_train, X_test = self.X[train_i], self.X[test_i]
            y_train = self.y[train_i]

            self.model.fit(X_train, y_train)
            predictions[test_i] = self.model.predict(X_test)
        return predictions
