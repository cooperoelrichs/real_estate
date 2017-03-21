from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import (KFold, cross_val_score)
from real_estate.models.model_analysis import ModelAnalysis


class GridSearch():
    def do(data_file_path, outputs_dir, xy_class, model_class, params):
        data = ModelAnalysis.read_data(data_file_path)
        xy = xy_class(data, exclude_suburb=False)

        model = model_class(
            xy.X.values, xy.y.values,
            xy.X.columns.values, data,
        )

        grid = GridSearchCV(
            model.model,
            params,
            scoring='mean_squared_error',
            cv=KFold(xy.y.values.shape[0]),
            verbose=0,
            n_jobs=4
        )

        grid.fit(xy.X.values, xy.y.values)
        model.model = grid.best_estimator_
        return grid.best_params_, model
