clear all 

log using mexicohousingdata.txt, text replace 

import delimited "/Users/maximianne/PycharmProjects/IndependentStudy/mex_housing_data.csv"

//fixes 
replace popularity = 3 if place_name == "Yucatán"
replace popularity = 3 if place_name == "Bucerías"
replace popularity = 1 if place_name == "Cuernavaca"
replace popularity = 2 if place_name == "Veracruz"
replace popularity = 3 if place_name == "Tulum"
replace popularity = 3 if place_name == "Playa del Carmen"
replace popularity = 3 if place_name == "San Miguel de Allende"
replace popularity = 3 if place_name == "Tepic"
replace popularity = 3 if place_name == "Cancún"
replace popularity = 3 if place_name == "San Pedro Garza García"
replace popularity = 3 if place_name == "Santa Catarina"
replace popularity = 1 if place_name == "Villa Lomas Altas 2a Secc."

//popularity
summarize popularity
summarize price_aprox_usd if popularity == 3
summarize price_aprox_usd if popularity == 2
summarize price_aprox_usd if popularity == 1

//dummy variables for popularity 
gen pop3 = 1 if popularity == 3
replace pop3 = 0 if pop3 == .

gen pop2 = 1 if popularity == 2
replace pop2 = 0 if pop2 == .

gen pop1 = 1 if popularity == 1
replace pop1 = 0 if pop1 == . 

gen house = 1 if property_type == 1
replace house =0 if house == .

sort year

gen log_price = ln(price_aprox_usd)

summarize log_price
summarize price_aprox_usd 
summarize usdmexval 
summarize rooms 
summarize house 
summarize surface_total_in_m2 
summarize pop1
summarize pop2
summarize pop3 

gen interactionPop1 = usdmexval*pop1 
gen interactionPop3 = usdmexval*pop3 

summarize interactionPop1
summarize interactionPop3


gen pricemex = price_aprox_usd*usdmexval
gen logpricemex = ln(pricemex)

regress log_price usdmexval rooms house surface_total_in_m2 pop3 interactionPop3

regress log_price usdmexval interactionPop1 rooms house surface_total_in_m2 pop1

log close 


