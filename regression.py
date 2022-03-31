# ---- Libraries ---#
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import geopy as geo
from geopy import distance


def latlon_to_tuple():
    lat_long_tuple_list = []
    lat_b = data['lat']
    lon_b = data['lon']
    length = data.shape[0]
    for i in range(0, length):
        (lat, lon) = (lat_b.iloc[i], lon_b.iloc[i])
        lat_long_tuple_list.append((lat, lon))
    return lat_long_tuple_list


def latlon_to_tuple2():
    lat_long_tuple_list = []
    lat_b = data2['LON']
    lon_b = data2['LAT']
    length = data2.shape[0]
    for i in range(0, length):
        (lat, lon) = (lat_b.iloc[i], lon_b.iloc[i])
        lat_long_tuple_list.append((lat, lon))
    return lat_long_tuple_list


def define_based_on_radius_3(list_main, list_check):
    # around_radius (int) – Radius in meters to search around the latitude/longitude.
    # Otherwise, a default radius is automatically computed given the area density.

    classification = []
    length_Main = len(list_main)
    # length_Main = data.shape[0]  # this is the length of the list all our data cities with
    # their coordinates and the distance
    length_check = len(list_check)
    # length_check = data2.shape[0]  # this is the length of the list of our defined "popular"
    # cities their coordinates and the distance
    for i in range(0, length_Main):
        check = list_main[i]
        # print("The value being checked:", check)
        count = 0
        for j in range(0, length_check):
            count += 1
            check_from = list_check[j]
            # print("The radius to be,", check_from)
            radius = data2['radius_3 '].iloc[j]
            radius2 = data2['radius_2'].iloc[j]
            dis = distance.distance(check_from, check).km
            if dis <= radius:
                # print("The radius,", check, "was in the bounds of ",radius)
                classification.append(3)
                break
            elif dis <= radius2:
                # print("The radius,", check, "was in the bounds of ",radius2)
                classification.append(2)
                break
            elif count == length_check:
                classification.append(1)
    return classification


def check_Nan_cities(lat_lon, cityList):
    areNan = []
    for i in lat_lon:
        if i == 0:
            areNan.append(cityList[i])

    result = []
    for i in areNan:
        if i not in result:
            result.append(i)

    return result


# classify cities
def getCities(column):
    cityList = []
    # append
    for i in column:
        if i not in cityList:
            cityList.append(i)
    return cityList


def give_citiesNum(list):
    count = 1
    for i in list:
        data['place_name'].replace({i: count}, inplace=True)
        count += 1


def create_currencyVal(main1, currency):
    currency_val = []
    for i in range(0, len(main1)):
        cur = main1.iloc[i]
        yearM = pd.to_datetime(cur['created_on']).year
        monthM = pd.to_datetime(cur['created_on']).month
        for j in range(0, len(currency)):
            check = currency.iloc[j]
            check2 = check['Date']
            check3 = check2.split("-")
            yearC = int(check3[0])
            monthC = int(check3[1])
            if yearM == yearC and monthM == monthC:
                value = check["Average"]
                currency_val.append(value)
                break
    return currency_val


if __name__ == "__main__":
    # CSV file which contains prices for different areas in Mexico
    data = pd.read_csv("properati_mx_2016_11_01_properties_sell.csv")
    data2 = pd.read_csv("cities_tier1_2.csv")
    data3 = pd.read_csv("mexico_currency_val_monthly.csv")

    '''Fixing the data:'''
    # print(list(data2.columns))
    data['created_on'] = pd.to_datetime(data['created_on'], format='%Y-%m-%d')

    data2['radius_3 '] = data2['radius_3 '].astype('int')
    data2['radius_2'] = data2['radius_2'].astype('int')

    data3.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    data3.rename(columns={'Month': 'Date'}, inplace=True)

    # make sure they are numerical values and delete nans
    data.dropna(subset=["price_aprox_usd"], inplace=True)
    data["year"] = pd.to_datetime(data['created_on']).dt.year
    print(data["year"])
    data.dropna(subset=["surface_total_in_m2"], inplace=True)
    data.dropna(subset=["rooms"], inplace=True)
    data.drop(columns=['description', 'operation', 'place_with_parent_names', 'geonames_id', 'currency',
                       'price_aprox_local_currency', 'price', 'surface_covered_in_m2', 'image_thumbnail', 'title',
                       'price_usd_per_m2', 'price_per_m2', 'floor', 'expenses', 'properati_url', 'lat_lon'],
              axis=1, inplace=True)

    # print(data.price_aprox_usd.to_string(index=False))
    data['price_aprox_usd'] = pd.to_numeric(data['price_aprox_usd'], downcast="float")
    # print(data.surface_total_in_m2.to_string(index=False))
    data['surface_total_in_m2'] = pd.to_numeric(data['surface_total_in_m2'], downcast="float")

    # property type
    # house = 1
    # apartment = 2
    data['property_type'].replace({"house": "1"}, inplace=True)
    data['property_type'].replace({"apartment": "2"}, inplace=True)
    # we want to drop the stores
    data['property_type'].replace({"store": np.nan}, inplace=True)
    data['property_type'].replace({"PH": np.nan}, inplace=True)
    data.dropna(subset=["property_type"], inplace=True)
    data['property_type'] = pd.to_numeric(data['property_type'], downcast="float")
    print(list(data.columns))
    # location in mexico:
    cityListALL = data['place_name']
    cities = getCities(cityListALL)  # gives each city a number
    # print(len(cities))  # there are 1556 different cities
    # print(len(cityListALL))  # there are 71,083 data with cities

    # getting long and lat into the proper one
    data['lat'] = data['lat'].fillna(0)
    data['lon'] = data['lon'].fillna(0)
    # check which cities had the NaN Val and Correct
    # print(check_Nan_cities(lat, cityListALL))
    # --> Nuevo Centre Urbano
    # 20.6296 -87.0739
    data['lat'].replace({0: 20.6296}, inplace=True)
    data['lon'].replace({0: -87.0739}, inplace=True)

    # replace these vals
    # converted to tuples

    latLonList_main = latlon_to_tuple()
    # print(latLonList_main)

    latLonList_cities = latlon_to_tuple2()
    # print(latlon_to_tuple2())

    classify = define_based_on_radius_3(latLonList_main, latLonList_cities)

    # print(classify)
    data['popularity'] = classify
    # print(data)
    data.drop(columns=['lat', 'lon', 'location'], axis=1, inplace=True)

    USD2MXNVal = create_currencyVal(data, data3)
    data["USDMEXVal"] = USD2MXNVal


    data.to_csv('mex_housing_data.csv')
