import pickle
import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import prepare_dataset

X_train, X_test, y_train, y_test = prepare_dataset.get_train_test_data()
params = yaml.safe_load(open("params.yaml"))['train']

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.12, 0.15, 0.18, 0.2],
    'max_depth': [1, 2, 3, 4, 5, 7, 9],
    'n_estimators': [10, 20, 35, 50, 75, 100, 200],
    'max_features': ['auto', None]
}
FILE_NAME = params['model_file_name']
seed = params["seed"]
np.random.seed(seed)

model_gbt = GradientBoostingRegressor(random_state=seed, verbose=1)


def grid_search_wrapper():
    skf = StratifiedKFold(n_splits=2)
    grid_search = GridSearchCV(model_gbt, param_grid, scoring='neg_mean_absolute_error', refit=True,
                               cv=skf, return_train_score=True, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print('################################----DONE----################################')
    return grid_search


#
grid_search = grid_search_wrapper()
pickle.dump(grid_search, open(FILE_NAME, "wb"))

print("Model saved successfully to {}".format(FILE_NAME))
