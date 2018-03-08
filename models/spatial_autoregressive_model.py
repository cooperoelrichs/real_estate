import numpy as np

from pysal.spreg.ml_error import ML_Error
from pysal.weights.weights import WSP
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

from real_estate.models.price_model import PriceModel


class SAR(PriceModel):
    HAS_SIMPLE_COEFS = True
    HAS_FEATURE_IMPORTANCE = False

    PARAMS = {
        'n_neighbors': 10,
        'radius': 1000,
        'use_radius': True,
        'method': 'full',
        'spat_diag': False,  # (boolean) – if True, include spatial diagnostics
    }

    def __init__(self, X, y, X_labels, points, params=None):
        if params == None:
            self.model = self.make_model(self.PARAMS)
        else:
            self.model = self.make_model(params)
        self.points = points
        self.setup_self(X, y, X_labels)

    def make_model(self, params):
        return KNearestNeighborsSpatialAutoregressiveModel(**params)


class KNearestNeighborsSpatialAutoregressiveModel(object):
    def __init__(self, n_neighbors, radius, use_radius, method, spat_diag):
        self.model_class = ML_Error
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.use_radius = use_radius
        self.method = method
        self.spat_diag = spat_diag

    def fit(self, X, y, points):
        if self.use_radius:
            self.nearest_neighbors = NearestNeighbors(
                radius=self.radius
            )
            self.nearest_neighbors.fit(points)
            w = self.nearest_neighbors.radius_neighbors_graph(
                points, mode='distance'
            )
        else:
            self.nearest_neighbors = NearestNeighbors(
                n_neighbors=self.n_neighbors + 1
            )
            self.nearest_neighbors.fit(points)
            w = self.nearest_neighbors.kneighbors_graph(
                points, mode='connectivity'
            )

        self.set_diag_to_zero(w)
        self.fitted_model = self.model_class(
            x=X, y=self.reshape(y), w=WSP(sparse=w).to_W(),
            method=self.method, spat_diag=self.spat_diag
        )
        print(
            'Fitment results: spatial autoregressive coefficient %.3f, R squared value %.3f.'
            % (self.fitted_model.lam, self.fitted_model.pr2)
        )

    def predict(self, X_pred, points):
        # w_pred = self.nearest_neighbors.kneighbors_graph(points)
        # y_pred = self.fitted_model.predict(X_pred, w_pred, points)

        c = self.fitted_model.betas[0, 0]
        b = self.fitted_model.betas[1:-1, :]
        y_pred = c + np.dot(X_pred, b)
        return y_pred

    def score(self, X_test, y_test, points):
        y_pred = self.predict(X_test, points)
        s = r2_score(y_test, y_pred)
        print('Score %.3f.' % s)
        return s

    def reshape(self, y):
        return np.array([y]).T

    def set_diag_to_zero(self, w):
        for i in range(w.shape[0]):
            w[i, i] = 0
        w.eliminate_zeros()
        return w

    # def points(self, X):
    #     return X['X', 'Y'].values.T
