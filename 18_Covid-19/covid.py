"""
ANSWERING the AD-HOC QUESTIONS

This module is COVID_BASE IMPLEMENTATION TO BUILD UP THE BASE DATA FRAMES to STORE ALL DATA (Confirmed, Fatalities, Recovered) 
from ALL COUNTRIES FROM FIRST DAY to LATEST DAY
"""
from covid_prep import *

''' 
All countries CONFIRMED_cases changed by day - Total Confirmed by Day in all countries
'''
sum_countries_confirmed = countries_confirmed.apply(lambda x: sum(x), axis=1)

'''
All Countries FATALITIES_cases changed by day - Total Deaths by Day in all countries
'''
sum_countries_deaths = countries_fatalities.apply(lambda x: sum(x), axis=1)
   
'''
All Countries RECOVERED_cases changed by day - Total Recovered by Day in all countries
'''
sum_countries_recovered = countries_recovered.apply(lambda x: sum(x), axis=1)

# Total Sum of Confirmed, Fatal, Recovered
totalConfirmed = countries_confirmed.tail(1).values.sum()
totalFatal = countries_fatalities.tail(1).values.sum()
totalRecovered = countries_recovered.tail(1).values.sum()

# Total all confirmed cases in all countries changed by day 
# totalconfirmed_by_day = [sum(data[day]['Confirmed']) for day in x_dat]
totalconfirmed_by_day = countries_confirmed.sum(axis=1).tolist()

# New Increasing/ changes cases in all countries changed by day
newCasesByDay = [totalconfirmed_by_day[0]]+[totalconfirmed_by_day[day+1]-totalconfirmed_by_day[day] for day in x_dat[:-1]]

# Total all fatalities cases in all countries changed by day
# totalfatalities_by_day = [sum(data[day]['Deaths']) for day in x_dat]
totalfatalities_by_day = countries_fatalities.sum(axis=1).tolist()

# Total all recovered cases in all countries changed by day
totalrecovered_by_day = countries_recovered.sum(axis=1).tolist()

# New Increasing/ changes Fatalities in ALL COUNTRIES changed by day
newConfirmedByDay = [totalconfirmed_by_day[0]] + [totalconfirmed_by_day[day+1] - totalconfirmed_by_day[day] for day in x_dat[:-1]]

# New Increasing/ changes Fatalities in ALL COUNTRIES changed by day
newFatalitiesByDay = [totalfatalities_by_day[0]] + [totalfatalities_by_day[day+1] - totalfatalities_by_day[day] for day in x_dat[:-1]]

# New Increasing/ changes Recovered in ALL COUNTRIES changed by day
newRecoveredByDay = [totalrecovered_by_day[0]] + [totalrecovered_by_day[day+1] - totalrecovered_by_day[day] for day in x_dat[:-1]]

num_infected_countries = [len(np.unique(data[day][data[day]['Confirmed']>0].filter(regex=("Country.*")).values)) for day in x_dat]

# Compute changes (confirmed, fatalities, recovered) by day in every country
countries_confirmed_by_day = pd.DataFrame(columns=countries_confirmed.columns)

for country in countries_confirmed.columns:
    new_values = [countries_confirmed[country].iloc[0]] + [countries_confirmed[country].iloc[day+1]-countries_confirmed[country].iloc[day] for day in x_dat[:-1]]
    countries_confirmed_by_day[country] = new_values

countries_fatalities_by_day =  pd.DataFrame(columns=countries_fatalities.columns) 
for country in countries_fatalities_by_day.columns:
    new_values = [countries_fatalities[country].iloc[0]] + [countries_fatalities[country].iloc[day+1]-countries_fatalities[country].iloc[day] for day in x_dat[:-1]]
    countries_fatalities_by_day[country] = new_values

countries_recovered_by_day = pd.DataFrame(columns=countries_recovered.columns)
for country in countries_recovered_by_day.columns:
    new_values = [countries_recovered[country].iloc[0]] + [countries_recovered[country].iloc[day+1]-countries_recovered[country].iloc[day] for day in x_dat[:-1]]
    countries_recovered_by_day[country] = new_values