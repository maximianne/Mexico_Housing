# -------Libraries ----------#
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# -------Functions & what they do----------#
# function to match the x variables to the y variables which are our months --> value
def x_var(years):
    amount = years * 12
    array = [0] * amount
    count = 1
    for i in range(amount):
        if count <= 12:
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


def matrix_avg_min_max(year_list):
    count = 1
    avg_min_max = np.empty((12, 3))
    i = 0
    while count <= 12:
        vals = month_list_fun(year_list, count)
        min = np.amin(vals)
        max = np.amax(vals)
        val = np.sum(vals)
        val_avg = val / len(vals)
        avg_min_max[i, 0] = val_avg
        avg_min_max[i, 1] = min
        avg_min_max[i, 2] = max
        count += 1
        i += 1
    return avg_min_max


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


def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')


def check_rsme_linear():
    root_mean_error = []
    score = []
    C_value = []
    i = 0.1
    dataf = pd.DataFrame(avgMinMaxAll)
    dataf.insert(0, "Month", x, True)
    dataf.rename(columns={0: 'Average', 1: 'Minimum', 2: 'Maximum'}, inplace=True)
    dataf.drop(columns=['Month'], axis=1, inplace=True)
    dataf['Month'] = dates
    pred_days = 6
    dataf['Prediction'] = dataf[['Average']].shift(-pred_days)

    x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    x_val = x_val[:len(dataf) - pred_days]

    y_val = np.array(dataf['Prediction'])
    y_val = y_val[:-pred_days]

    train_x, test_x, train_y, test_y = train_test_split(x_val, y_val, test_size=0.2)

    while i <= 100:
        C_value.append(i)
        svr = SVR(kernel='linear', C=i)
        svr.fit(train_x, train_y)

        score_l = svr.score(test_x, test_y)

        x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
        pred = svr.predict(x_val)

        rsme_l = mean_squared_error(x_val, pred, squared=False)
        root_mean_error.append(rsme_l)
        print("Root mean:                        ", rsme_l)
        score.append(score_l)
        print("score:         ", score_l)
        i += 0.1
        print("count:", i)
    return root_mean_error, score, C_value


def check_rsme_poly(deg):
    root_mean_error = []
    score = []
    C_value = []
    i = 0.1
    dataf = pd.DataFrame(avgMinMaxAll)
    dataf.insert(0, "Month", x, True)
    dataf.rename(columns={0: 'Average', 1: 'Minimum', 2: 'Maximum'}, inplace=True)
    dataf.drop(columns=['Month'], axis=1, inplace=True)
    dataf['Month'] = dates
    pred_days = 6
    dataf['Prediction'] = dataf[['Average']].shift(-pred_days)

    x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    x_val = x_val[:len(dataf) - pred_days]

    y_val = np.array(dataf['Prediction'])
    y_val = y_val[:-pred_days]

    train_x, test_x, train_y, test_y = train_test_split(x_val, y_val, test_size=0.2)

    while i <= 100:
        C_value.append(i)
        svr = SVR(kernel='poly', degree=deg, C=i)
        svr.fit(train_x, train_y)

        score_l = svr.score(test_x, test_y)

        x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
        pred = svr.predict(x_val)

        rsme_l = mean_squared_error(x_val, pred, squared=False)
        root_mean_error.append(rsme_l)
        print("Root mean:                        ", rsme_l)
        score.append(score_l)
        print("score:         ", score_l)
        i += 0.1
        print("count:", i)
    return root_mean_error, score, C_value


def check_rsme_rbf():
    root_mean_error = []
    score = []
    C_value = []
    i = 0.1
    dataf = pd.DataFrame(avgMinMaxAll)
    dataf.insert(0, "Month", x, True)
    dataf.rename(columns={0: 'Average', 1: 'Minimum', 2: 'Maximum'}, inplace=True)
    dataf.drop(columns=['Month'], axis=1, inplace=True)
    dataf['Month'] = dates
    pred_days = 6
    dataf['Prediction'] = dataf[['Average']].shift(-pred_days)

    x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    x_val = x_val[:len(dataf) - pred_days]

    y_val = np.array(dataf['Prediction'])
    y_val = y_val[:-pred_days]

    train_x, test_x, train_y, test_y = train_test_split(x_val, y_val, test_size=0.2)

    while i <= 100:
        C_value.append(i)
        svr = SVR(kernel='rbf', C=i)
        svr.fit(train_x, train_y)

        score_l = svr.score(test_x, test_y)

        x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
        pred = svr.predict(x_val)

        rsme_l = mean_squared_error(x_val, pred, squared=False)
        root_mean_error.append(rsme_l)
        print("Root mean:                        ", rsme_l)
        score.append(score_l)
        print("score:         ", score_l)
        i += 0.1
        print("count:", i)
    return root_mean_error, score, C_value


def check_rsme_sig():
    root_mean_error = []
    score = []
    C_value = []
    i = 0.1
    dataf = pd.DataFrame(avgMinMaxAll)
    dataf.insert(0, "Month", x, True)
    dataf.rename(columns={0: 'Average', 1: 'Minimum', 2: 'Maximum'}, inplace=True)
    dataf.drop(columns=['Month'], axis=1, inplace=True)
    dataf['Month'] = dates
    pred_days = 6
    dataf['Prediction'] = dataf[['Average']].shift(-pred_days)

    x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    x_val = x_val[:len(dataf) - pred_days]

    y_val = np.array(dataf['Prediction'])
    y_val = y_val[:-pred_days]

    train_x, test_x, train_y, test_y = train_test_split(x_val, y_val, test_size=0.2)

    while i <= 100:
        C_value.append(i)
        svr = SVR(kernel='sigmoid', C=i)
        svr.fit(train_x, train_y)

        score_l = svr.score(test_x, test_y)

        x_val = np.array(dataf.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
        pred = svr.predict(x_val)

        rsme_l = mean_squared_error(x_val, pred, squared=False)
        root_mean_error.append(rsme_l)
        print("Root mean:                        ", rsme_l)
        score.append(score_l)
        print("score:         ", score_l)
        i += 0.1
        print("count:", i)
    return root_mean_error, score, C_value


def check_rmse_rbfH():
    root_mean_error = []
    score = []
    C_value = []
    i = 0.1
    dataf = pd.read_csv("Home_price.csv")
    dataf['DATE'] = pd.to_datetime(dataf['DATE'].str.strip(), format='%Y/%m/%d')
    pred_days = 2
    dataf['Prediction'] = dataf[['QMXR628BIS']].shift(-pred_days)
    x_val = np.array(dataf.drop(columns=['DATE', 'Prediction']))
    x_val = x_val[:len(dataf) - pred_days]

    y_val = np.array(dataf['Prediction'])
    y_val = y_val[:-pred_days]

    train_x, test_x, train_y, test_y = train_test_split(x_val, y_val, test_size=0.3)

    while i <= 100:
        C_value.append(i)
        svr = SVR(kernel='rbf', C=i)
        svr.fit(train_x, train_y)
        score_l = svr.score(test_x, test_y)
        score.append(score_l)
        svr_rbf.fit(train_x, train_y)
        x_val = np.array(dataf.drop(columns=['DATE', 'Prediction']))
        pred_every = svr_rbf.predict(x_val)
        rsme_l = mean_squared_error(x_val, pred_every, squared=False)
        root_mean_error.append(rsme_l)
        print("Root mean:                        ", rsme_l)
        print("score:         ", score_l)
        i += 0.1
        print("count:", i)

    return root_mean_error, score, C_value


def check_rmse_linearH():
    root_mean_error = []
    score = []
    C_value = []
    i = 0.1
    dataf = pd.read_csv("Home_price.csv")
    dataf['DATE'] = pd.to_datetime(dataf['DATE'].str.strip(), format='%Y/%m/%d')
    pred_days = 2
    dataf['Prediction'] = dataf[['QMXR628BIS']].shift(-pred_days)
    x_val = np.array(dataf.drop(columns=['DATE', 'Prediction']))
    x_val = x_val[:len(dataf) - pred_days]

    y_val = np.array(dataf['Prediction'])
    y_val = y_val[:-pred_days]

    train_x, test_x, train_y, test_y = train_test_split(x_val, y_val, test_size=0.3)

    while i <= 100:
        C_value.append(i)
        svr = SVR(kernel='linear', C=i)
        svr.fit(train_x, train_y)
        score_l = svr.score(test_x, test_y)
        svr_rbf.fit(train_x, train_y)
        x_val = np.array(dataf.drop(columns=['DATE', 'Prediction']))
        pred_every = svr_rbf.predict(x_val)
        rsme_l = mean_squared_error(x_val, pred_every, squared=False)
        root_mean_error.append(rsme_l)
        print("Root mean:                        ", rsme_l)
        score.append(score_l)
        print("score:         ", score_l)
        i += 0.1
        print("count:", i)

    return root_mean_error, score, C_value


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
    avg_1995 = avg_month(year_1995)
    avg_1996 = avg_month(year_1996)
    avg_1997 = avg_month(year_1997)
    avg_1998 = avg_month(year_1998)
    avg_1999 = avg_month(year_1999)
    avg_2000 = avg_month(year_2000)
    avg_2001 = avg_month(year_2001)
    avg_2002 = avg_month(year_2002)
    avg_2003 = avg_month(year_2003)
    avg_2004 = avg_month(year_2004)
    avg_2005 = avg_month(year_2005)
    avg_2006 = avg_month(year_2006)
    avg_2007 = avg_month(year_2007)
    avg_2008 = avg_month(year_2008)
    avg_2009 = avg_month(year_2009)
    avg_2010 = avg_month(year_2010)
    avg_2011 = avg_month(year_2011)
    avg_2012 = avg_month(year_2012)
    avg_2013 = avg_month(year_2013)
    avg_2014 = avg_month(year_2014)
    avg_2015 = avg_month(year_2015)
    avg_2016 = avg_month(year_2016)
    avg_2017 = avg_month(year_2017)
    avg_2018 = avg_month(year_2018)
    avg_2019 = avg_month(year_2019)
    avg_2020 = avg_month(year_2020)
    avg_2021 = avg_month(year_2021)

    # this is the average + min + max of that month
    avgMinMax_1994 = matrix_avg_min_max(year_1994)
    avgMinMax_1995 = matrix_avg_min_max(year_1995)
    avgMinMax_1996 = matrix_avg_min_max(year_1996)
    avgMinMax_1997 = matrix_avg_min_max(year_1997)
    avgMinMax_1998 = matrix_avg_min_max(year_1998)
    avgMinMax_1999 = matrix_avg_min_max(year_1999)
    avgMinMax_2000 = matrix_avg_min_max(year_2000)
    avgMinMax_2001 = matrix_avg_min_max(year_2001)
    avgMinMax_2002 = matrix_avg_min_max(year_2002)
    avgMinMax_2003 = matrix_avg_min_max(year_2003)
    avgMinMax_2004 = matrix_avg_min_max(year_2004)
    avgMinMax_2005 = matrix_avg_min_max(year_2005)
    avgMinMax_2006 = matrix_avg_min_max(year_2006)
    avgMinMax_2007 = matrix_avg_min_max(year_2007)
    avgMinMax_2008 = matrix_avg_min_max(year_2008)
    avgMinMax_2009 = matrix_avg_min_max(year_2009)
    avgMinMax_2010 = matrix_avg_min_max(year_2010)
    avgMinMax_2011 = matrix_avg_min_max(year_2011)
    avgMinMax_2012 = matrix_avg_min_max(year_2012)
    avgMinMax_2013 = matrix_avg_min_max(year_2013)
    avgMinMax_2014 = matrix_avg_min_max(year_2014)
    avgMinMax_2015 = matrix_avg_min_max(year_2015)
    avgMinMax_2016 = matrix_avg_min_max(year_2016)
    avgMinMax_2017 = matrix_avg_min_max(year_2017)
    avgMinMax_2018 = matrix_avg_min_max(year_2018)
    avgMinMax_2019 = matrix_avg_min_max(year_2019)
    avgMinMax_2020 = matrix_avg_min_max(year_2020)
    avgMinMax_2021 = matrix_avg_min_max(year_2021)

    avgMinMaxAll = np.vstack(
        (avgMinMax_1994, avgMinMax_1995, avgMinMax_1996, avgMinMax_1997, avgMinMax_1998, avgMinMax_1999,
         avgMinMax_2000, avgMinMax_2001, avgMinMax_2002, avgMinMax_2003, avgMinMax_2004, avgMinMax_2005,
         avgMinMax_2006, avgMinMax_2007, avgMinMax_2008, avgMinMax_2009, avgMinMax_2010, avgMinMax_2011,
         avgMinMax_2012, avgMinMax_2013, avgMinMax_2014, avgMinMax_2015, avgMinMax_2016, avgMinMax_2017,
         avgMinMax_2018, avgMinMax_2019, avgMinMax_2020, avgMinMax_2021))

    # convert to the same base year price
    # avgMinMaxAll = basePrice(avgMinMaxAll)
    # print(avgMinMaxAll)

    x = x_var(28)
    x = np.array(x)

    date_1994 = get_year_list(1994)
    date_1995 = get_year_list(1995)
    date_1996 = get_year_list(1996)
    date_1997 = get_year_list(1997)
    date_1998 = get_year_list(1998)
    date_1999 = get_year_list(1999)
    date_2000 = get_year_list(2000)
    date_2001 = get_year_list(2001)
    date_2002 = get_year_list(2002)
    date_2003 = get_year_list(2003)
    date_2004 = get_year_list(2004)
    date_2005 = get_year_list(2005)
    date_2006 = get_year_list(2006)
    date_2007 = get_year_list(2007)
    date_2008 = get_year_list(2008)
    date_2009 = get_year_list(2009)
    date_2010 = get_year_list(2010)
    date_2011 = get_year_list(2011)
    date_2012 = get_year_list(2012)
    date_2013 = get_year_list(2013)
    date_2014 = get_year_list(2014)
    date_2015 = get_year_list(2015)
    date_2016 = get_year_list(2016)
    date_2017 = get_year_list(2017)
    date_2018 = get_year_list(2018)
    date_2019 = get_year_list(2019)
    date_2020 = get_year_list(2020)
    date_2021 = get_year_list(2021)

    dates = np.concatenate(
        (date_1994, date_1995, date_1996, date_1997, date_1998, date_1999, date_2000, date_2001, date_2002,
         date_2003, date_2004, date_2005, date_2006, date_2007, date_2008, date_2009, date_2010, date_2011,
         date_2012, date_2013, date_2014, date_2015, date_2016, date_2017, date_2018, date_2019, date_2020,
         date_2021), axis=None)

    # SVC stuff
    ''' DECIDE WHAT VALUE FOR C TO USE FOR CURRENCY EXCHANGE'''
    # Linear
    '''rsme, scores, c_list = check_rsme_linear()
    pyplot.title('RSME for Different C values- Linear Kernel')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values - Linear Kernel')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()

    # Polynomial
    rsme, scores, c_list = check_rsme_poly(2)
    pyplot.title('RSME for Different C values- Polynomial Kernel, Deg = 2')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values - Polynomial Kernel, Deg = 2')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()'''

    '''rsme, scores, c_list = check_rsme_poly(3)
    pyplot.title('RSME for Different C values- Polynomial Kernel, Deg = 3')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values - Polynomial Kernel, Deg = 3')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()

    # RBF
    rsme, scores, c_list = check_rsme_rbf()
    pyplot.title('RSME for Different C values- RBF Kernel')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values - RBF Kernel')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()

    # Sigmoid
    rsme, scores, c_list = check_rsme_sig()
    pyplot.title('RSME for Different C values- Sigmoid Kernel')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values - Sigmoid Kernel')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()'''

    # SVR
    df = pd.DataFrame(avgMinMaxAll)
    df.insert(0, "Month", x, True)
    df.rename(columns={0: 'Average', 1: 'Minimum', 2: 'Maximum'}, inplace=True)
    df.drop(columns=['Month'], axis=1, inplace=True)
    df['Month'] = dates
    df.to_csv('mexico_currency_val_monthly.csv')

    prediction_days = 6
    df['Prediction'] = df[['Average']].shift(-prediction_days)

    X = np.array(df.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    X = X[:len(df) - prediction_days]

    Y = np.array(df['Prediction'])
    Y = Y[:-prediction_days]

    x_m = makeListNum(len(df['Month']))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    prediction_days_array = np.array(df.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction'], axis=1))[
                            -prediction_days:]
    rsme_all = []
    scores_all = []

    """ Linear Kernel"""
    svr_linear = SVR(kernel='linear',C =100 )
    svr_linear.fit(X_train, y_train)

    svr_Linear_confidence = svr_linear.score(X_test, y_test)
    print("Our predicted accuracy for currency exchange using Linear Kernel is:", svr_Linear_confidence * 100)
    scores_all.append(svr_Linear_confidence)
    # the models predicted values for the test data
    svm_predictionTestDataLinear = svr_linear.predict(X_test)

    # our model prediction for the next 12 month
    svm_predictionLinear = svr_linear.predict(prediction_days_array)

    # the actual values of the next 12 months
    n = prediction_days = 6
    df.tail(prediction_days)

    X = np.array(df.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    predict_everythingLinear = svr_linear.predict(X)

    linear_rsme = mean_squared_error(X, predict_everythingLinear, squared=False)
    rsme_all.append(linear_rsme)

    """Polynomial Kernel Degree = 2"""
    svr_polynomial2 = SVR(kernel='poly', degree=2, C= 0.1)
    svr_polynomial2.fit(X_train, y_train)

    svr_poly2_confidence = svr_polynomial2.score(X_test, y_test)
    print("Our predicted accuracy for currency exchange using Polynomial degree = 2 kernel is:",
          svr_poly2_confidence * 100)
    scores_all.append(svr_poly2_confidence)

    # the models predicted values for the test data
    svm_predictionTestDataPoly2 = svr_polynomial2.predict(X_test)
    # print(len(svm_predictionTestData))

    # our model prediction for the next 12 month
    svm_predictionPoly2 = svr_polynomial2.predict(prediction_days_array)

    # the actual values of the next 12 months
    n = prediction_days = 6
    df.tail(prediction_days)

    X = np.array(df.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    predict_everythingPoly2 = svr_polynomial2.predict(X)

    poly2_rsme = mean_squared_error(X, predict_everythingPoly2, squared=False)
    rsme_all.append(poly2_rsme)

    """Polynomial Kernel Degree = 3"""
    svr_polynomial3 = SVR(kernel='poly', degree=3, C= 0.1)
    svr_polynomial3.fit(X_train, y_train)

    svr_poly3_confidence = svr_polynomial3.score(X_test, y_test)
    print("Our predicted accuracy for currency exchange using Polynomial degree = 3 kernel is:",
          svr_poly3_confidence * 100)
    scores_all.append(svr_poly3_confidence)

    # the models predicted values for the test data
    svm_predictionTestDataPoly3 = svr_polynomial3.predict(X_test)
    # print(len(svm_predictionTestData))

    # our model prediction for the next 12 month
    svm_predictionPoly3 = svr_polynomial3.predict(prediction_days_array)

    # the actual values of the next 12 months
    n = prediction_days = 6
    df.tail(prediction_days)

    X = np.array(df.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    predict_everythingPoly3 = svr_polynomial3.predict(X)
    poly3_rsme = mean_squared_error(X, predict_everythingPoly3, squared=False)
    rsme_all.append(poly3_rsme)

    """ RBF Kernel"""
    svr_rbf = SVR(kernel='rbf', C=100)
    svr_rbf.fit(X_train, y_train)
    svr_rbf_confidence = svr_rbf.score(X_test, y_test)
    print("Our predicted accuracy for currency exchange using RBF is:", svr_rbf_confidence * 100)
    scores_all.append(svr_rbf_confidence)
    # the models predicted values for the test data
    svm_predictionTestData = svr_rbf.predict(X_test)
    # print(len(svm_predictionTestData))

    # our model prediction for the next 12 month
    svm_prediction = svr_rbf.predict(prediction_days_array)

    # the actual values of the next 12 months
    n = prediction_days = 6
    df.tail(prediction_days)

    X = np.array(df.drop(columns=['Month', 'Minimum', 'Maximum', 'Prediction']))
    predict_everything = svr_rbf.predict(X)

    rbf_rsme = mean_squared_error(X, predict_everything, squared=False)
    rsme_all.append(rbf_rsme)

    pyplot.title('Currency Exchange USD, Peso Avg 1994-2021')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Price (USD)', fontsize=8)
    pyplot.plot(x_m, df['Average'], label="Actual data", linewidth=2)
    pyplot.plot(x_m, predict_everythingLinear, label="Linear")
    pyplot.plot(x_m, predict_everythingPoly2, label="Polynomial Degree = 2")
    pyplot.plot(x_m, predict_everythingPoly3, label="Polynomial Degree = 3")
    pyplot.plot(x_m, predict_everything, label="RBF")
    pyplot.legend()
    pyplot.show()

    x_rsmes = [0, 1, 2, 3]
    label_rsme = ["Linear", "Poly D=2", "Poly D = 3", "RBF"]
    pyplot.title('Root Sqaure Mean Error by Kernel')
    pyplot.xlabel('Type of Kernel', fontsize=10)
    pyplot.ylabel('RSME value', fontsize=8)
    pyplot.xticks(x_rsmes, label_rsme)
    pyplot.bar(x_rsmes, rsme_all)
    pyplot.show()

    x_rsmes = [0, 1, 2, 3]
    label_rsme = ["Linear", "Poly D=2", "Poly D = 3", "RBF"]
    pyplot.title('Scores by Kernel')
    pyplot.xlabel('Type of Kernel', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.xticks(x_rsmes, label_rsme)
    pyplot.bar(x_rsmes, scores_all)
    pyplot.show()

    # y_predNew = svr_rbf.predict(next_6_days)
    # print(y_predNew)

    # -------HOUSING PRICE INDEX--------- #

    '''rsme, scores, c_list = check_rmse_rbfH()
    pyplot.title('RSME for Different C values- RBF')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values- RBF')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()

    rsme, scores, c_list = check_rmse_linearH()
    pyplot.title('RSME for Different C values- Linear')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('RSME', fontsize=8)
    pyplot.plot(c_list, rsme)
    pyplot.show()

    pyplot.title('Scores for Different C values- Linear')
    pyplot.xlabel('C value', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.plot(c_list, scores)
    pyplot.show()'''


    df2 = pd.read_csv("Home_price.csv")
    df2['DATE'] = pd.to_datetime(df2['DATE'].str.strip(), format='%Y/%m/%d')

    prediction_days = 2
    df2['Prediction'] = df2[['QMXR628BIS']].shift(-prediction_days)

    X = np.array(df2.drop(columns=['DATE', 'Prediction']))
    X = X[:len(df2) - prediction_days]

    Y = np.array(df2['Prediction'])
    Y = Y[:-prediction_days]

    x_m = makeListNum(len(df2['DATE']))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    prediction_days_array = np.array(df2.drop(columns=['DATE', 'Prediction'], axis=1))[
                            -prediction_days:]
    # print(prediction_days_array)
    rsme_all = []
    scores_all = []

    """LINEAR"""
    svr_rbf = SVR(kernel='linear', C=70)
    svr_rbf.fit(X_train, y_train)

    svr_rbf_confidence = svr_rbf.score(X_test, y_test)
    scores_all.append(svr_rbf_confidence)
    # print("Our predicted accuracy for housing price index is:", svr_rbf_confidence * 100)

    # the models predicted values for the test data
    svm_predictionTestData = svr_rbf.predict(X_test)
    # print(len(svm_predictionTestData))

    # the actual values of the currency
    # print(y_test)

    # our model prediction for the next 2 quarters
    svm_prediction = svr_rbf.predict(prediction_days_array)
    # print(svm_prediction)

    # the actual values of the next 2 quarters
    n = prediction_days = 2
    df2.tail(prediction_days)

    X = np.array(df2.drop(columns=['DATE', 'Prediction']))
    predict_everything = svr_rbf.predict(X)

    linear_rsme = mean_squared_error(X, predict_everything, squared=False)
    rsme_all.append(linear_rsme)

    """RBF"""

    svr_rbf = SVR(kernel='rbf', C=40)
    svr_rbf.fit(X_train, y_train)

    svr_rbf_confidence = svr_rbf.score(X_test, y_test)
    # print("Our predicted accuracy for housing price index is:", svr_rbf_confidence * 100)
    scores_all.append(svr_rbf_confidence)
    # the models predicted values for the test data
    svm_predictionTestData = svr_rbf.predict(X_test)
    # print(len(svm_predictionTestData))

    # the actual values of the currency
    # print(y_test)

    # our model prediction for the next 2 quarters
    svm_prediction = svr_rbf.predict(prediction_days_array)
    # print(svm_prediction)

    # the actual values of the next 2 quarters
    n = prediction_days = 2
    df2.tail(prediction_days)

    X = np.array(df2.drop(columns=['DATE', 'Prediction']))
    predict_everythingRBF = svr_rbf.predict(X)

    rbf_rsme = mean_squared_error(X, predict_everythingRBF, squared=False)
    rsme_all.append(rbf_rsme)

    '''pyplot.title('Housing price index- Quarterly from 2005-2021')
    pyplot.xlabel('Quarters', fontsize=10)
    pyplot.ylabel('Price (USD)', fontsize=8)
    pyplot.plot(x_m, df2['QMXR628BIS'])
    pyplot.show()'''

    pyplot.title('Housing price index- Quarterly from 2005- 2021')
    pyplot.xlabel('Quarters', fontsize=10)
    pyplot.ylabel('Price (USD)', fontsize=8)
    pyplot.plot(x_m, df2['QMXR628BIS'], label="Actual data", linewidth=2)
    pyplot.plot(x_m, predict_everything, label="Linear")
    pyplot.plot(x_m, predict_everythingRBF, label="RBF")
    pyplot.legend()
    pyplot.show()

    x_rsmes = [0, 1]
    label_rsme = ["Linear", "RBF"]
    pyplot.title('Root Sqaure Mean Error by Kernel')
    pyplot.xlabel('Type of Kernel', fontsize=10)
    pyplot.ylabel('RSME value', fontsize=8)
    pyplot.xticks(x_rsmes, label_rsme)
    pyplot.bar(x_rsmes, rsme_all)
    pyplot.show()

    x_rsmes = [0, 1]
    label_rsme = ["Linear", "RBF"]
    pyplot.title('Scores by Kernel')
    pyplot.xlabel('Type of Kernel', fontsize=10)
    pyplot.ylabel('Score', fontsize=8)
    pyplot.xticks(x_rsmes, label_rsme)
    pyplot.bar(x_rsmes, scores_all)
    pyplot.show()




