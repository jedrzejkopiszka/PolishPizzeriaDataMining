import prepate_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

X_y, modeling_validation, train_test = prepate_dataset.get_train_test_data()

#baseline model
def basline_model(y_test,X_train,X_test,y_train):
    baseline_test = y_test.to_frame()
    baseline_test['predicted_pizza_count'] = 0
    X_train['pizza_count'] = y_train.values
    for index, row in X_test.iterrows():
        baseline_test.loc[index, 'predicted_pizza_count'] = \
            X_train.loc[X_train.weekday == row.weekday].pizza_count.mean()
    baseline_test['difference'] = baseline_test.apply(lambda x: abs(x['predicted_pizza_count'] - x['pizza_count']),
                                                      axis=1)

    print("MAE calculated \"by handy\": ", baseline_test.difference.mean())
    print("Sum of errors: ", baseline_test.difference.sum())
    print("MSE: ", mean_squared_error(y_test, baseline_test.predicted_pizza_count))
    print("MAE using built-in formula: ", mean_absolute_error(y_test, baseline_test.predicted_pizza_count))

