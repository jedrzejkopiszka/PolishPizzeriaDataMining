import pickle
import yaml
from sklearn.metrics import mean_absolute_error
from src import prepare_dataset
import json


X_train, X_test, y_train, y_test = prepare_dataset.get_train_test_data()

params = yaml.safe_load(open("../params.yaml"))['evaluate']
MODEL_FILE_NAME = params['model_file_name']
RESULTS_FILE_NAME = params['results_file_name']
loaded_model = pickle.load(open(MODEL_FILE_NAME, 'rb'))


prediction_model = loaded_model.predict(X_test)
mse = mean_absolute_error(y_test,prediction_model)
best_params = loaded_model.best_params_

results = {
    "file_name":MODEL_FILE_NAME,
    "used_model":str(loaded_model.estimator),
    "best_parameters":best_params,
    "mse":mse,
}
RESULTS_FILE_NAME = "{}_{}".format(str(loaded_model.estimator).split('(')[0],RESULTS_FILE_NAME)
with open(RESULTS_FILE_NAME,"w") as results_file:
    json.dump(results,results_file,indent = 2)
print("Results saved successfully to {}".format(RESULTS_FILE_NAME))
print(json.dumps(results,indent=2))