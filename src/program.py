from datetime import datetime
import pickle
import numpy as np
from src import prepare_dataset

dates = ["01-01", "06-01", "01-05", "03-05", "15-08",
         "01-11", "11-11", "25-12", "26-12"]

dates_2016 = ["27-03", "28-03", "15-05", "26-05"]
dates_2017 = ["16-04", "17-04", "04-06", "15-06"]

dates_years = [dates_2016, dates_2017]

years = ["2016", "2017"]
free_days = [[False for x in range(366)], [False for x in range(365)]]

for x, year in enumerate(years):
    dates_loop = dates + dates_years[x]
    for date in dates_loop:
        date = date + "-" + year
        free_days[x][datetime.strptime(date, "%d-%m-%Y").timetuple().tm_yday - 1] = True
pizza_daily_copy = prepare_dataset.get_pizza_data()
print(type(pizza_daily_copy))
d = dict()
for day_of_week in range(7):
    d[day_of_week] = np.mean(pizza_daily_copy.loc[pizza_daily_copy.weekday == day_of_week].pizza_count)


def get_savings(model, test_X, test_Y, price_overevaluation, price_underevaluation):
    pizza_sales_model = model.predict(test_X)
    suma = 0
    suma1 = 0
    for idx, weekday in enumerate(test_X.weekday):
        difference_model = pizza_sales_model[idx] - test_Y.values[idx]
        difference_average = d[weekday] - test_Y.values[idx]

        signs = [1 if difference_model > 0 else -1 if difference_model < 0 else 0,
                 1 if difference_average > 0 else -1 if difference_average < 0 else 0]

        suma += (abs(difference_average) * (price_overevaluation if signs[1] == 1 else price_underevaluation)) - (
                abs(difference_model) * (price_overevaluation if signs[0] == 1 else price_underevaluation))

    print("Savings: {} in {} days".format(suma, len(test_X)))


X_train, X_test, y_train, y_test = prepare_dataset.get_train_test_data()

model = pickle.load(open('model.sav', 'rb'))

get_savings(model, X_test, y_test, 15, 30)
