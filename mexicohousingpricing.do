clear all 

log using mexicohousingdata.txt, text replace 

import delimited "/Users/maximianne/PycharmProjects/IndependentStudy/mex_housing_data.csv"

//fixes 
drop if price_aprox_usd > 1000000
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
replace popularity = 2 if popularity == 1 & price_aprox_usd > 600000 

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

// summary statistics 
by year: summarize price_aprox_usd usdmexval rooms house surface_total_in_m2 if popularity ==3 

by year: summarize price_aprox_usd usdmexval rooms house surface_total_in_m2 if popularity ==2 

by year: summarize price_aprox_usd usdmexval rooms house surface_total_in_m2 if popularity ==1

summarize price_aprox_usd

summarize usdmexval

summarize rooms

summarize surface_total_in_m2

summarize house 

summarize popularity


summarize price_aprox_usd if popularity == 3

summarize usdmexval if popularity == 3

summarize rooms if popularity == 3

summarize house  if popularity == 3

summarize surface_total_in_m2 if popularity == 3

summarize pop3 


summarize price_aprox_usd if popularity == 2

summarize usdmexval if popularity == 2

summarize rooms if popularity == 2

summarize house  if popularity == 2

summarize surface_total_in_m2 if popularity == 2

summarize pop2


summarize price_aprox_usd if popularity == 1

summarize usdmexval if popularity == 1

summarize rooms if popularity == 1

summarize house  if popularity == 1

summarize surface_total_in_m2 if popularity == 1

summarize pop1


regress price_aprox_usd usdmexval rooms house surface_total_in_m2 pop3

regress price_aprox_usd usdmexval rooms house surface_total_in_m2 pop2

regress price_aprox_usd usdmexval rooms house surface_total_in_m2 pop1

regress price_aprox_usd usdmexval rooms house surface_total_in_m2 popularity

log close 


