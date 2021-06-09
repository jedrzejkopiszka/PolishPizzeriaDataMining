import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit


def get_train_test_data():
    pizza_data = pd.read_csv('../data/pizza_data.csv')
    pizza_data.drop(pizza_data.loc[(pizza_data.year == 2016) & (pizza_data.month == 10) & (pizza_data.day == 11)].index,
                    inplace=True)
    pizza_data = pizza_data.rename(columns={'count': 'pizza_count'})
    pizza_data = pizza_data.drop(columns=['weekend_day'])  # Weekend_day can be dropped - 0 entropy
    pizza_daily = pd.DataFrame(columns=pizza_data.columns)
    for index, row in pizza_data.iterrows():
        if int(row.hour) == 0:
            pizza_daily = pizza_daily.append(row)
        elif pizza_daily.size > 0:
            pizza_daily.iloc[-1, 0] += row.pizza_count
    pizza_daily = pizza_daily.set_index('new_index')  # We rename index

    pizza_daily = pizza_daily.drop(columns=['hour'])  # Hour is no longer valid as it is daily data
    pizza_daily = pizza_daily[['year', 'month', 'day', 'working_day', 'public_holiday', 'pizza_count']]
    pizza_daily['weekday'] = pizza_daily.apply(
        lambda x: datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)
    pizza_daily_copy = pizza_daily.copy()
    pizza_daily_copy = pizza_daily_copy[['year', 'month', 'day', 'weekday',
                                         'working_day', 'public_holiday', 'pizza_count']]

    x, y = pizza_daily_copy.iloc[:, :-1], pizza_daily_copy.pizza_count
    # We want to split data into 3 parts - train, test, validation and use validation only for testing the final model
    X_for_modeling, X_for_validation, y_for_modeling, y_for_validation = train_test_split(x, y, shuffle=False,
                                                                                          test_size=61)

    # In this step we split data for modelling to find best algorithm, tune parameters and assess performance
    X_train, X_test, y_train, y_test = train_test_split(X_for_modeling, y_for_modeling, shuffle=False, test_size=61)
    return (
        (x, y), (X_for_modeling, X_for_validation, y_for_modeling, y_for_validation),
        (X_train, X_test, y_train, y_test))
