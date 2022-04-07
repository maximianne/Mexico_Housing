# -------Libraries ----------#
import np as np
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# -------Functions & what they do----------#
# function to match the x variables to the y variables which are our months --> value
def x_varQ(years):
    amount = years * 4
    array = [0] * amount
    count = 1
    for i in range(amount):
        if count <= 4:
            array[i] = count
            count += 1
        else:
            array[i] = 1
            count = 2
    return array


# function to extract the data for a single month given a year, remove values with 0
def month_list_fun(year_list, num):
    vals = year_list[year_list['DATE'].dt.month == num]
    to_return = vals.loc[:, 'DEXMXUS']
    # remove values that are 0

    return to_return


# this takes in a year and creates a list for the average price each month
def avg_month(year_list):
    count = 1
    year_avg = [0] * 12
    i = 0
    while count <= 12:
        vals = month_list_fun(year_list, count)
        val = np.sum(vals)
        val_avg = val / len(vals)
        year_avg[i] = val_avg
        count += 1
        i += 1
    return year_avg


def makeListNum(num):
    n = []
    for i in range(0, num):
        n.append(i + 1)
    n = np.array(n)
    return n


def get_year_list(year):
    month = 1
    list = []
    for i in range(0, 12):
        if month < 10:
            string = str(year) + "-0" + str(month) + "-01"
            list.append(string)
            month += 1
        else:
            string = str(year) + "-" + str(month) + "-01"
            list.append(string)
            month += 1
    return list


def quarterly_avg(year):
    split = np.split(year, 4)
    quarter = []
    for i in range(0, len(split)):
        avg = 0
        arr = split[i]
        for j in range(0, len(arr)):
            avg += arr[j]
        quarter.append(avg / 3)
    return quarter


def makesingle_array(array):
    array_r = []
    for i in range(0, len(array)):
        arrayC = array[i]
        array_r.append(arrayC[0])
    returns = np.array(array_r)
    return returns


if __name__ == "__main__":
    # USD to Pesos CSV file
    data = pd.read_csv("USD_to_Peso.csv")
    # format this correctly: we want to formate one column as the date
    # we want to convert the numbers into floats, so we can process them
    data['DATE'] = pd.to_datetime(data['DATE'].str.strip(), format='%Y/%m/%d')
    data['DEXMXUS'].replace({".": "0"}, inplace=True)
    data['DEXMXUS'] = pd.to_numeric(data['DEXMXUS'], downcast="float")
    # ensures that the dates which the DEXMXUS val is 0 is
    # dropped to avoid the min getting messed up
    target = data[data['DEXMXUS'] > 0]

    # separate data by year
    year_1994 = target[target['DATE'].dt.year == 1994]
    year_1995 = target[target['DATE'].dt.year == 1995]
    year_1996 = target[target['DATE'].dt.year == 1996]
    year_1997 = target[target['DATE'].dt.year == 1997]
    year_1998 = target[target['DATE'].dt.year == 1998]
    year_1999 = target[target['DATE'].dt.year == 1999]
    year_2000 = target[target['DATE'].dt.year == 2000]
    year_2001 = target[target['DATE'].dt.year == 2001]
    year_2002 = target[target['DATE'].dt.year == 2002]
    year_2003 = target[target['DATE'].dt.year == 2003]
    year_2004 = target[target['DATE'].dt.year == 2004]
    year_2005 = target[target['DATE'].dt.year == 2005]
    year_2006 = target[target['DATE'].dt.year == 2006]
    year_2007 = target[target['DATE'].dt.year == 2007]
    year_2008 = target[target['DATE'].dt.year == 2008]
    year_2009 = target[target['DATE'].dt.year == 2009]
    year_2010 = target[target['DATE'].dt.year == 2010]
    year_2011 = target[target['DATE'].dt.year == 2011]
    year_2012 = target[target['DATE'].dt.year == 2012]
    year_2013 = target[target['DATE'].dt.year == 2013]
    year_2014 = target[target['DATE'].dt.year == 2014]
    year_2015 = target[target['DATE'].dt.year == 2015]
    year_2016 = target[target['DATE'].dt.year == 2016]
    year_2017 = target[target['DATE'].dt.year == 2017]
    year_2018 = target[target['DATE'].dt.year == 2018]
    year_2019 = target[target['DATE'].dt.year == 2019]
    year_2020 = target[target['DATE'].dt.year == 2020]
    year_2021 = target[target['DATE'].dt.year == 2021]

    # make average of months for each year by month
    avg_1994 = avg_month(year_1994)
    avg_1994 = np.array(avg_1994)
    avg_1994 = quarterly_avg(avg_1994)

    avg_1995 = avg_month(year_1995)
    avg_1995 = np.array(avg_1995)
    avg_1995 = quarterly_avg(avg_1995)

    avg_1996 = avg_month(year_1996)
    avg_1996 = np.array(avg_1996)
    avg_1996 = quarterly_avg(avg_1996)

    avg_1997 = avg_month(year_1997)
    avg_1997 = np.array(avg_1997)
    avg_1997 = quarterly_avg(avg_1997)

    avg_1998 = avg_month(year_1998)
    avg_1998 = np.array(avg_1998)
    avg_1998 = quarterly_avg(avg_1998)

    avg_1999 = avg_month(year_1999)
    avg_1999 = np.array(avg_1999)
    avg_1999 = quarterly_avg(avg_1999)

    avg_2000 = avg_month(year_2000)
    avg_2000 = np.array(avg_2000)
    avg_2000 = quarterly_avg(avg_2000)

    avg_2001 = avg_month(year_2001)
    avg_2001 = np.array(avg_2001)
    avg_2001 = quarterly_avg(avg_2001)

    avg_2002 = avg_month(year_2002)
    avg_2002 = np.array(avg_2002)
    avg_2002 = quarterly_avg(avg_2002)

    avg_2003 = avg_month(year_2003)
    avg_2003 = np.array(avg_2003)
    avg_2003 = quarterly_avg(avg_2003)

    avg_2004 = avg_month(year_2004)
    avg_2004 = np.array(avg_2004)
    avg_2004 = quarterly_avg(avg_2004)

    avg_2005 = avg_month(year_2005)
    avg_2005 = np.array(avg_2005)
    avg_2005 = quarterly_avg(avg_2005)

    avg_2006 = avg_month(year_2006)
    avg_2006 = np.array(avg_2006)
    avg_2006 = quarterly_avg(avg_2006)

    avg_2007 = avg_month(year_2007)
    avg_2007 = np.array(avg_2007)
    avg_2007 = quarterly_avg(avg_2007)

    avg_2008 = avg_month(year_2008)
    avg_2008 = np.array(avg_2008)
    avg_2008 = quarterly_avg(avg_2008)

    avg_2009 = avg_month(year_2009)
    avg_2009 = np.array(avg_2009)
    avg_2009 = quarterly_avg(avg_2009)

    avg_2010 = avg_month(year_2010)
    avg_2010 = np.array(avg_2010)
    avg_2010 = quarterly_avg(avg_2010)

    avg_2011 = avg_month(year_2011)
    avg_2011 = np.array(avg_2011)
    avg_2011 = quarterly_avg(avg_2011)

    avg_2012 = avg_month(year_2012)
    avg_2012 = np.array(avg_2012)
    avg_2012 = quarterly_avg(avg_2012)

    avg_2013 = avg_month(year_2013)
    avg_2013 = np.array(avg_2013)
    avg_2013 = quarterly_avg(avg_2013)

    avg_2014 = avg_month(year_2014)
    avg_2014 = np.array(avg_2014)
    avg_2014 = quarterly_avg(avg_2014)

    avg_2015 = avg_month(year_2015)
    avg_2015 = np.array(avg_2015)
    avg_2015 = quarterly_avg(avg_2015)

    avg_2016 = avg_month(year_2016)
    avg_2016 = np.array(avg_2016)
    avg_2016 = quarterly_avg(avg_2016)

    avg_2017 = avg_month(year_2017)
    avg_2017 = np.array(avg_2017)
    avg_2017 = quarterly_avg(avg_2017)

    avg_2018 = avg_month(year_2018)
    avg_2018 = np.array(avg_2018)
    avg_2018 = quarterly_avg(avg_2018)

    avg_2019 = avg_month(year_2019)
    avg_2019 = np.array(avg_2019)
    avg_2019 = quarterly_avg(avg_2019)

    avg_2020 = avg_month(year_2020)
    avg_2020 = np.array(avg_2020)
    avg_2020 = quarterly_avg(avg_2020)

    avg_2021 = avg_month(year_2021)
    avg_2021 = np.array(avg_2021)
    avg_2021 = quarterly_avg(avg_2021)

    avgAll = np.concatenate(
        (avg_1994, avg_1995, avg_1996, avg_1997, avg_1998, avg_1999,
         avg_2000, avg_2001, avg_2002, avg_2003, avg_2004, avg_2005,
         avg_2006, avg_2007, avg_2008, avg_2009, avg_2010, avg_2011,
         avg_2012, avg_2013, avg_2014, avg_2015, avg_2016, avg_2017,
         avg_2018, avg_2019, avg_2020, avg_2021))

    x = x_varQ(28)
    x = np.array(x)

    # SVC stuff
    df = pd.DataFrame(avgAll)
    df.rename(columns={0: 'Average'}, inplace=True)
    # print(df)

    prediction_days = 8
    df['Prediction'] = df[['Average']].shift(-prediction_days)

    X = np.array(df.drop(columns=['Prediction']))
    X = X[:len(df) - prediction_days]

    Y = np.array(df['Prediction'])
    Y = Y[:-prediction_days]

    x_m = makeListNum(len(df['Average']))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    prediction_days_array = np.array(df.drop(columns=['Prediction'], axis=1))[
                            -prediction_days:]

    svr_linearCurrency = SVR(kernel='linear', C=100)
    svr_linearCurrency.fit(X_train, y_train)

    svr_Linear_confidence = svr_linearCurrency.score(X_test, y_test)
    # print("Our predicted accuracy for currency exchange using Linear Kernel is:", svr_Linear_confidence * 100)
    # the models predicted values for the test data
    svm_predictionTestDataLinear = svr_linearCurrency.predict(X_test)

    # our model prediction for the next 12 month
    svm_predictionLinear = svr_linearCurrency.predict(prediction_days_array)
    print("our model predicition for the next 3 years:", svm_predictionLinear)

    # the actual values of the next 12 months
    n = prediction_days = 8
    print(df.tail(prediction_days))

    X_actuallyCurrency = np.array(df.drop(columns=['Prediction']))
    predict_everythingLinearCurrency = svr_linearCurrency.predict(X_actuallyCurrency)

    predict_currency_2005 = predict_everythingLinearCurrency[45:]

    linear_rsme = mean_squared_error(X_actuallyCurrency, predict_everythingLinearCurrency, squared=False)
    # print("The linear rsme is: ", linear_rsme)

    next_8CURRENCY = df.tail(prediction_days)

    # -------HOUSING PRICE INDEX--------- #
    df2 = pd.read_csv("Home_price.csv")
    df2['DATE'] = pd.to_datetime(df2['DATE'].str.strip(), format='%Y/%m/%d')

    prediction_days = 8
    df2['Prediction'] = df2[['QMXR628BIS']].shift(-prediction_days)

    X = np.array(df2.drop(columns=['DATE', 'Prediction']))
    X = X[:len(df2) - prediction_days]

    Y = np.array(df2['Prediction'])
    Y = Y[:-prediction_days]

    x_m = makeListNum(len(df2['DATE']))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    prediction_days_arrayHOUSING = np.array(df2.drop(columns=['DATE', 'Prediction'], axis=1))[
                                   -prediction_days:]

    svr_linearHousing = SVR(kernel='linear', C=70)
    svr_linearHousing.fit(X_train, y_train)

    svr_rbf_confidence = svr_linearHousing.score(X_test, y_test)
    # print("Our predicted accuracy for housing price index is:", svr_rbf_confidence * 100)

    # the models predicted values for the test data
    svm_predictionTestData = svr_linearHousing.predict(X_test)
    # print(len(svm_predictionTestData))

    # the actual values of the currency
    # print(y_test)

    # our model prediction for the next 2 quarters
    svm_prediction = svr_linearHousing.predict(prediction_days_array)
    print(svm_prediction)

    # the actual values of the next 2 quarters
    n = prediction_days = 8
    df2.tail(prediction_days)

    X_actual_housing = np.array(df2.drop(columns=['DATE', 'Prediction']))
    predict_everythingLinearHousing = svr_linearHousing.predict(X_actual_housing)

    linear_rsme = mean_squared_error(X_actual_housing, predict_everythingLinearHousing, squared=False)
    # print("The linear rsme is: ", linear_rsme)

    housing = X_actual_housing
    currency = X_actuallyCurrency[45:]

    svr_linearHousing.predict(prediction_days_array)
    print(df2.tail(prediction_days))
    pyplot.title('Housing Price Index and Currency - Quarterly from 2005- 2021')
    pyplot.xlabel('Quarters', fontsize=10)
    pyplot.ylabel('Price (USD)/ Peso (MXN)', fontsize=8)
    pyplot.plot(x_m, currency, label="Currency (MXN)")
    pyplot.plot(x_m, np.multiply(.3, housing), label="Housing Index (Scaled down by .3) (USD)")
    pyplot.annotate("COVID", xy=(60.5, 30))
    plt.axvline(x=60, color='r')
    plt.axvline(x=68, color='r')
    pyplot.legend()
    pyplot.show()

    graph_originalCURRENCY = X_actuallyCurrency[len(X_actuallyCurrency) - 16:]
    print(graph_originalCURRENCY)
    graph_originalHOUSING = X_actual_housing[len(X_actual_housing) - 16:]
    print(graph_originalHOUSING)

    next_8HOUSING = X_actual_housing[len(X_actual_housing) - 8:]
    next_8CURRENCY = X_actuallyCurrency[len(X_actuallyCurrency) - 8:]

    next_twoyearsCurrency = svr_linearCurrency.predict(prediction_days_array)
    next_8C = makesingle_array(next_8CURRENCY)

    next_twoyearsHousing = svr_linearHousing.predict(prediction_days_arrayHOUSING)
    next_8H = makesingle_array(next_8HOUSING)
    zero_array = [np.NAN, np.NAN, np.NAN, np.NAN, np.NAN, np.NAN, np.NAN, np.NAN]
    both_currency = np.concatenate((zero_array, next_twoyearsCurrency))
    print(len(both_currency))
    both_housing = np.concatenate((zero_array, next_twoyearsHousing))
    pred_house = np.multiply(.2, both_housing)
    x_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pyplot.title("Forecasting 2020-2021")
    pyplot.xlabel('Quarters', fontsize=10)
    pyplot.ylabel('Price (USD)/ Peso (MXN)', fontsize=8)
    plt.axvline(x=9, color='r')
    pyplot.annotate("Forecasting period", xy=(10.5, 24))
    pyplot.plot(x_m, graph_originalCURRENCY, label="Actual Trend: Currency (MXN)")
    pyplot.plot(x_m, both_currency, label="Forecast Currency (MXN)")
    pyplot.plot(x_m, np.multiply(.2, graph_originalHOUSING), label="Actual Trend: Housing (.2) (USD)")
    pyplot.plot(x_m, pred_house, label="Forecast Housing (.2) (USD)")
    pyplot.legend()
    pyplot.show()


