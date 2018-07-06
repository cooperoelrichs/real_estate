import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error


class PriceModel(object):
    def __init__(self, X, y, X_labels, params):
        if params is None:
            self.model = self.MODEL_CLASS(**self.PARAMS)
        else:
            self.model = self.MODEL_CLASS(**params)
        self.setup_self(X, y, X_labels)

    def setup_self(self, X, y, X_labels):
        self.X = X
        self.y = y
        self.X_labels = X_labels
        # self.categorical_groups = categorical_groups
        # self.by_categorical_groups = by_categorical_groups
        # self.df = df
        # self.seed = 1

    def mean_absolute_error(self, y_pred):
        return mean_absolute_error(y_true=self.y, y_pred=y_pred)

    def model_summary(self):
        print('This model does not support printing a model summary.')

    def show_live_results(self, outputs_folder, name):
        print('This model does not support showing live results.')

    def cv_score_and_predict(self, n_folds):
        if n_folds is None:
            n_folds = self.N_FOLDS

        scores = []
        mean_absolute_errors = []
        estimates = np.zeros_like(self.y)
        test = np.ones_like(self.y, dtype=bool)

        folds = self.construct_folds(n_folds)
        for i, indicies in enumerate(folds):
            train_i, test_i = indicies
            score = self.score(train_i, test_i)
            scores += [score]

            estimates[test_i] = self.model.predict(self.X[test_i])
            test[test_i] = False

            mae = np.mean(np.abs(self.y[test_i] - estimates[test_i]))
            mean_absolute_errors += [mae]

            print('    Fold %i: %.4f, %.0f' % (i, score, mae))

        scores = np.array(scores)
        mean_absolute_errors = np.array(mean_absolute_errors)

        if (n_folds != 1) and test.any():
            raise RuntimeError(
                'Estimates check failed, %i failures' % np.sum(test)
            )

        return scores, mean_absolute_errors, estimates

    def scores(self, n_folds):
        scores = []
        folds = self.construct_folds(n_folds)
        for i, indicies in enumerate(folds):
            train_i, test_i = indicies
            score = self.score(train_i, test_i)
            print('    Fold %i: %f' % (i, score))
            scores += [score]

        scores = np.array(scores)
        return scores

    def construct_folds(self, n_folds):
        if n_folds == 1:
            folds = list(KFold(n=self.X.shape[0], n_folds=5))[:1]
        else:
            folds = KFold(n=self.X.shape[0], n_folds=n_folds)
        return folds

    def score(self, train_i, test_i):
        self.model.fit(self.X[train_i], self.y[train_i])
        score = self.model.score(self.X[test_i], self.y[test_i])

        if np.isnan(score) or score < -10 ** 6:
            raise RuntimeError(
                'Encounted a bad score, model likely unidentifiable'
            )

        return score

    def score_with_id_filter(self, train_i, test_i):
        id_filter = self.identifiability_filter(self.X[train_i])
        self.fit(self.X[:, id_filter][train_i], self.y[train_i])
        score = self.score(self.X[:, id_filter][test_i], self.y[test_i])
        return score

    def identifiability_filter(self, X_subset):
        # Ensure that there are no empty (all 0) columns, because the
        # parameter for such a column wouldn't be identifiable.
        empty_column_filter = X_subset.sum(axis=0) > 0

        # Every set of dummy variables must one row of all zeros, so that
        # the parameters for that dummy set are identifiable.
        categoricals_filter = np.ones((X_subset.shape[1]), dtype=bool)
        for name, i_first, i_last in self.categorical_groups:
            group_filter = np.zeros((X_subset.shape[1]), dtype=bool)

            group_filter[i_first:i_last+1] = True
            group_columns = X_subset[:, group_filter]  # & empty_column_filter]

            any_empty_rows = (group_columns.sum(axis=1) == 0).any()
            if not any_empty_rows:
                categoricals_filter[i_last - 1] = False

        # Every column of a Linear by Categorical variable set must contain
        # more than one unique value (other than zero) so that it isn't
        # identicle to the suburb dummy variable for the same suburb.
        by_categoricals_filter = np.ones((X_subset.shape[1]), dtype=bool)
        for name, pop, i_first, i_last in self.by_categorical_groups:
            for i in np.arange(i_first, i_last):
                uniques = np.unique(X_subset[:, i])
                uniques = uniques[(~ np.isnan(uniques)) & (uniques != 0)]
                num_uniques = uniques.shape[0]
                if num_uniques == 1:
                    # print(name, pop, uniques)
                    by_categoricals_filter[i] = False

        return (
            categoricals_filter &
            by_categoricals_filter &
            empty_column_filter
        )

    def cv_predict(self):
        raise NotImplementedError()
