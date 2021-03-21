"""
ANSWERING the AD-HOC QUESTIONS

This module is COVID_BASE IMPLEMENTATION TO BUILD UP THE BASE DATA FRAMES to STORE ALL DATA (Confirmed, Fatalities, Recovered) 
from ALL COUNTRIES FROM FIRST DAY to LATEST DAY
"""
from covid_import import *

'''
Meta_Data Preprocessing (Renaming, Key-Values, losing data) 
Goal: make data consistent before processing
'''
lastday=len(data)-1

# search all negative values in data or NaN and fill them all with 0 
data_date = dict([(i, files[i]) for i in range(len(files))]) 

new_days_list = np.arange(0, len(list(data_date.keys())[:-1]), 1)  # re-index from 0
for day in new_days_list:
    data[day][['Confirmed','Deaths','Recovered']] = data[day][['Confirmed','Deaths','Recovered']].mask(data[day][['Confirmed','Deaths','Recovered']]<0, 0)

countries = set()
infected_countries = dict()
for day in data_date.keys():
    infected_countries[day] = set(np.unique(data[day][data[day]['Confirmed']>0].filter(regex=("Country.*")).values))
    for c in infected_countries[day]:
        countries.add(c)

countries_confirmed = pd.DataFrame(columns=countries)
countries_fatalities = pd.DataFrame(columns=countries)
countries_recovered = pd.DataFrame(columns=countries)

for day in x_dat:   
    if 'Country/Region' in data[day].columns:
        countries_confirmed= countries_confirmed.append(data[day].groupby('Country/Region').sum().transpose().loc['Confirmed'], ignore_index=True)
        countries_fatalities= countries_fatalities.append(data[day].groupby('Country/Region').sum().transpose().loc['Deaths'], ignore_index=True)
        countries_recovered= countries_recovered.append(data[day].groupby('Country/Region').sum().transpose().loc['Recovered'], ignore_index=True)
    else:
        countries_confirmed= countries_confirmed.append(data[day].groupby('Country_Region').sum().transpose().loc['Confirmed'], ignore_index=True)
        countries_fatalities= countries_fatalities.append(data[day].groupby('Country_Region').sum().transpose().loc['Deaths'], ignore_index=True)
        countries_recovered= countries_recovered.append(data[day].groupby('Country_Region').sum().transpose().loc['Recovered'], ignore_index=True)
        
countries_confirmed.fillna(0, inplace=True)
countries_fatalities.fillna(0, inplace=True)
countries_recovered.fillna(0, inplace=True)

'''
Hard code for Korea, China, Hong Kong, and United Kingdom
Reasons:  
    - South Korea has three namings 'South Korea' and 'Korea, South', 'Republic of Korea'
    - China has two different namings ('Mainland, China' and 'China')
    - Hong Kong is removed from data after day 48 (start day 0), also naming as 'Hong Kong SAR' at day 48
    - Taiwan has two different namings ('Taiwan' and 'Taiwan*')
    - UK has two different namings ('UK', and 'United Kingdom')
    - Macau was given in the first days, but removed recently
'''
columns_removed = ['UK', 'Mainland China', 'Viet Nam', 'Republic of Korea', 'South Korea', 'Iran (Islamic Republic of)', 'Macao SAR'
, 'Bahamas, The', 'Gambia, The', 'Taiwan', 'Republic of Ireland', 'Russian Federation', 'Czech Republic', 'Diamond Princess', 'Cruise Ship', 'Others', 'Hong Kong SAR']

countries_confirmed['United Kingdom'] = countries_confirmed['United Kingdom'] + countries_confirmed['UK']
countries_confirmed['China'] = countries_confirmed['China'] + countries_confirmed['Mainland China']
countries_confirmed['Vietnam'] = countries_confirmed['Vietnam'] + countries_confirmed['Viet Nam']
countries_confirmed['Korea, South'] = countries_confirmed['Korea, South'] + countries_confirmed['Republic of Korea'] + countries_confirmed['South Korea'] 
countries_confirmed['Iran'] = countries_confirmed['Iran'] + countries_confirmed['Iran (Islamic Republic of)']
countries_confirmed['Macau'] = countries_confirmed['Macau'] + countries_confirmed['Macao SAR']
countries_confirmed['Bahamas'] = countries_confirmed['Bahamas'] + countries_confirmed['Bahamas, The']
countries_confirmed['Gambia'] = countries_confirmed['Gambia'] + countries_confirmed['Gambia, The']
countries_confirmed['Taiwan*'] = countries_confirmed['Taiwan*'] + countries_confirmed['Taiwan']
countries_confirmed['Hong Kong'] = countries_confirmed['Hong Kong'] + countries_confirmed['Hong Kong SAR']
countries_confirmed['Ireland'] = countries_confirmed['Ireland'] + countries_confirmed['Republic of Ireland']
countries_confirmed['Russia'] = countries_confirmed['Russia'] + countries_confirmed['Russian Federation']
countries_confirmed['Czechia'] = countries_confirmed['Czechia'] + countries_confirmed['Czech Republic']
countries_confirmed['Others_and_ships'] = countries_confirmed['Others'] + countries_confirmed['Cruise Ship'] + countries_confirmed['Diamond Princess']

countries_fatalities['United Kingdom'] = countries_fatalities['United Kingdom'] + countries_fatalities['UK']
countries_fatalities['China'] = countries_fatalities['China'] + countries_fatalities['Mainland China']
countries_fatalities['Vietnam'] = countries_fatalities['Vietnam'] + countries_fatalities['Viet Nam']
countries_fatalities['Korea, South'] = countries_fatalities['Korea, South'] + countries_fatalities['Republic of Korea'] + countries_fatalities['South Korea'] 
countries_fatalities['Iran'] = countries_fatalities['Iran'] + countries_fatalities['Iran (Islamic Republic of)']
countries_fatalities['Macau'] = countries_fatalities['Macau'] + countries_fatalities['Macao SAR']
countries_fatalities['Bahamas'] = countries_fatalities['Bahamas'] + countries_fatalities['Bahamas, The']
countries_fatalities['Gambia'] = countries_fatalities['Gambia'] + countries_fatalities['Gambia, The']
countries_fatalities['Taiwan*'] = countries_fatalities['Taiwan*'] + countries_fatalities['Taiwan']
countries_fatalities['Hong Kong'] = countries_fatalities['Hong Kong'] + countries_fatalities['Hong Kong SAR']
countries_fatalities['Ireland'] = countries_fatalities['Ireland'] + countries_fatalities['Republic of Ireland']
countries_fatalities['Russia'] = countries_fatalities['Russia'] + countries_fatalities['Russian Federation']
countries_fatalities['Czechia'] = countries_fatalities['Czechia'] + countries_fatalities['Czech Republic']
countries_fatalities['Others_and_ships'] = countries_fatalities['Others'] + countries_fatalities['Cruise Ship'] + countries_fatalities['Diamond Princess']

countries_recovered['United Kingdom'] = countries_recovered['United Kingdom'] + countries_recovered['UK']
countries_recovered['China'] = countries_recovered['China'] + countries_recovered['Mainland China']
countries_recovered['Vietnam'] = countries_recovered['Vietnam'] + countries_recovered['Viet Nam']
countries_recovered['Korea, South'] = countries_recovered['Korea, South'] + countries_recovered['Republic of Korea'] + countries_recovered['South Korea'] 
countries_recovered['Iran'] = countries_recovered['Iran'] + countries_recovered['Iran (Islamic Republic of)']
countries_recovered['Macau'] = countries_recovered['Macau'] + countries_recovered['Macao SAR']
countries_recovered['Bahamas'] = countries_recovered['Bahamas'] + countries_recovered['Bahamas, The']
countries_recovered['Gambia'] = countries_recovered['Gambia'] + countries_recovered['Gambia, The']
countries_recovered['Taiwan*'] = countries_recovered['Taiwan*'] + countries_recovered['Taiwan']
countries_recovered['Hong Kong'] = countries_recovered['Hong Kong'] + countries_recovered['Hong Kong SAR']
countries_recovered['Ireland'] = countries_recovered['Ireland'] + countries_recovered['Republic of Ireland']
countries_recovered['Russia'] = countries_recovered['Russia'] + countries_recovered['Russian Federation']
countries_recovered['Czechia'] = countries_recovered['Czechia'] + countries_recovered['Czech Republic']
countries_recovered['Others_and_ships'] = countries_recovered['Others'] + countries_recovered['Cruise Ship'] + countries_recovered['Diamond Princess']

'''
HARD-CODE: (SHOULD MOVE TO COVID_IMPORT)
Reason: US Recoveries have been nullified in the same data source, but distributed in other data source.
Need to hard-code to merge partly data together.
See: https://github.com/CSSEGISandData/COVID-19/issues/3464
'''
data_us_rec = [data_us[day]['Recovered'].sum() for day in x_dat_us]
data_us_recovered = pd.DataFrame(data={'Recovered': data_us_rec})
countries_recovered['US'][len(x_dat)-len(x_dat_us):len(x_dat)] = data_us_recovered['Recovered']

for column_rem in columns_removed:
    countries_confirmed.drop(column_rem, axis=1, inplace=True)
    countries_fatalities.drop(column_rem, axis=1, inplace=True)
    countries_recovered.drop(column_rem, axis=1, inplace=True)

# # collect all data into list all countries of tuple (country, confirmed), (country, fatalities), (country, recovered)
num_infected_countries = [len(infected_countries[day]) for day in x_dat]

# create dictionary of country's population
country_pop_dict = dict()
countries_no_infopopulation = list()

for country in countries_confirmed.columns:
    if country in sdata['Country_Region'].unique():
        country_pop_dict[country]=sdata[sdata['Country_Region']==country]['Population'].values[0]  
    else:
        countries_no_infopopulation.append(country)
        country_pop_dict[country]='NaN'

# print("\nNo Information about the population of these countries: \n", countries_no_infopopulation)

# Input population of the following countries manually  (year 2020)
# these are data estimated from Wikipedia, and https://www.worldometers.info/world-population/population-by-country/
# and https://worldpopulationreview.com/countries/
# and Cruise Ships  https://en.wikipedia.org/wiki/COVID-19_pandemic_on_cruise_ships#Ships_with_confirmed_cases_on_board
country_pop_dict['Hong Kong'] = 7500700
country_pop_dict['Republic of the Congo'] = 5381359
country_pop_dict['Aruba'] = 106314
country_pop_dict['Cayman Islands'] = 64948
country_pop_dict['Mayotte'] = 270372
country_pop_dict['Faroe Islands'] = 48678
country_pop_dict['Ivory Coast'] = 26378274
country_pop_dict['The Gambia'] = 2173999
country_pop_dict['Saint Martin'] = 38666
country_pop_dict['St. Martin'] = 38666
country_pop_dict['Cape Verde'] = 555987
country_pop_dict['Reunion'] = 859959
country_pop_dict[' Azerbaijan'] = 10139177
country_pop_dict['Greenland'] = 56770
country_pop_dict['Channel Islands'] = 173863
country_pop_dict['Guam'] = 168775
country_pop_dict['North Ireland'] = 1885000
country_pop_dict['Curacao'] = 157538
country_pop_dict['Vatican City'] = 801
country_pop_dict['The Bahamas'] = 393244
country_pop_dict['Martinique'] = 375265
country_pop_dict['Taipei and environs'] = 2646204 # wikipedia https://en.wikipedia.org/wiki/Taipei 2019
country_pop_dict['Palestine'] = 5101414
country_pop_dict['Saint Barthelemy'] = 9877
country_pop_dict['Mayotte'] = 272815
country_pop_dict['Gibraltar'] = 33691
country_pop_dict['Republic of Moldova'] = 4033963
country_pop_dict['Ivory Coast'] = 26867385
country_pop_dict['Macau'] = 682800
country_pop_dict['Guadeloupe'] = 400124
country_pop_dict['Puerto Rico'] = 2860853
country_pop_dict['French Guiana'] = 298682
country_pop_dict['Jersey'] = 173863  # Population of Channel Islands (2020 and historical)
country_pop_dict['occupied Palestinian territory'] = 0
country_pop_dict['Faroe Islands'] = 52154
country_pop_dict['Guernsey'] = 67052
country_pop_dict['East Timor'] = 1318445
country_pop_dict['Others_and_ships'] = 3711
