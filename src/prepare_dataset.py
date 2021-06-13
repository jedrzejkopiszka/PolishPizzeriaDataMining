import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
import yaml
import sys
import os
import io

params = yaml.safe_load(open("../params.yaml"))['prepare']

if len(sys.argv) != 1:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

TEST_SIZE = params["test_size"]


def add_unix(row):
    return pd.Timestamp(row['date']).timestamp()

def get_pizza_data():
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
    pizza_daily['new_index'] = list(np.arange(1, pizza_daily.shape[0] + 1, 1))

    pizza_daily = pizza_daily.set_index('new_index')  # We rename index

    pizza_daily = pizza_daily.drop(columns=['hour'])  # Hour is no longer valid as it is daily data
    pizza_daily = pizza_daily[['year', 'month', 'day', 'working_day', 'public_holiday', 'pizza_count']]
    pizza_daily = pizza_daily.astype(
        {"year": int, "month": int, "day": int, "working_day": int, "public_holiday": int, "pizza_count": int})
    pizza_daily['weekday'] = pizza_daily.apply(
        lambda x: datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)
    pizza_daily_copy = pizza_daily.copy()
    pizza_daily_copy = pizza_daily_copy.iloc[:-1]
    pizza_daily_copy = pizza_daily_copy[['year', 'month', 'day', 'weekday',
                                         'working_day', 'public_holiday', 'pizza_count']]
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 0), 'pizza_count'] = 63
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 1), 'pizza_count'] = 74
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 2), 'pizza_count'] = 112
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 3), 'pizza_count'] = 84
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 4), 'pizza_count'] = 127
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 5), 'pizza_count'] = 147
    pizza_daily_copy.loc[(pizza_daily_copy.pizza_count == 0) & (pizza_daily_copy.weekday == 6), 'pizza_count'] = 108
    return pizza_daily_copy

def get_train_test_data():
    pizza_daily_copy = get_pizza_data()
    X, y = pizza_daily_copy.iloc[:, :-1], pizza_daily_copy.pizza_count
    X_for_modeling, X_for_validation, y_for_modeling, y_for_validation = train_test_split(X, y, shuffle=False,
                                                                                          test_size=TEST_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X_for_modeling, y_for_modeling, shuffle=False, test_size=TEST_SIZE)
    return (X_train, X_test, y_train, y_test)
