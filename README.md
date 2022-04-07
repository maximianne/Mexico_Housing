# Mexico_Housing

CSV file contains the CSV files used for the project.

The Python file contains all the python files for the Project:

regression.py: 
The code that was used to clean up the housing cvs 

Basic clean up: 
- price_aprox_usd: 
NaN (missing) values were dropped 
converted into a numerical value: float 
- rooms: 
NaN (missing) values were dropped 
- surface_total_in_m2
NaN (missing) values were dropped 
converted into a numerical value: float 
- property_type:
Only apartment and house were used, all else were dropped, house = 1 and apartment = 2
Further modified as house, if house =1 it is a house, if house = 0 then it is an apartment (STATA) 
- rooms:
The number of rooms the property has 
- place_name:
The name of the city which the property is located at 
- lon, and lat:
These were used to classify and create the variable: “popularity” where the latitude and longitude coordinates were used to check if they were within the radius of “popular” cities in Mexico where tourists visit often.
To choose which cities were the most popular, various articles were used to classify. First popular cities were taken from basic travel articles from travel.news and planetware, after, coastal cities were added into the list from travelmexicosolo. Using freemaptools the radius was determined. If the city is within the first radius, meaning it is within the city limits, then the city gets a score of 3, if it is within the second radius then the city gets a score of 2, everything else gets a score of 1. 
Popularity variable was modified via STATA if certain areas were missed. 
- usdmexval: (ADDED) 
Added the value of 1 USD in MXN according to the date of listing, taken from 

The STATA file contains the STATA do file as well as the log for the regression analysis of this project
