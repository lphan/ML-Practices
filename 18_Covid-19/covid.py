"""
ANSWERING the AD-HOC QUESTIONS

This module is COVID_BASE IMPLEMENTATION TO BUILD UP THE BASE DATA FRAMES to STORE ALL DATA (Confirmed, Fatalities, Recovered) 
from ALL COUNTRIES FROM FIRST DAY to LATEST DAY
"""
from covid_import import *

'''
Data Preprocessing 
'''
lastday=len(data)-1

# search all negative values in data or NaN and fill them all with 0 
data_date = dict([(i, files[i]) for i in range(len(files))]) 

new_days_list = np.arange(0, len(list(data_date.keys())[:-1]), 1)  # re-index from 0
for day in new_days_list:
    data[day][['Confirmed','Deaths','Recovered']] = data[day][['Confirmed','Deaths','Recovered']].mask(data[day][['Confirmed','Deaths','Recovered']]<0, 0)

# collect all data into list all countries of tuple (country, confirmed), (country, fatalities), (country, recovered)
countries = sdata['Country_Region'].unique()

''' Number of all infected countries changed by day '''
# early country data value is South Korea, later country data value is Korea, South
infected_countries_earliest = np.unique(data[0][data[0]['Confirmed']>0].filter(regex=("Country.*")).values)

infected_countries_latest = np.unique(data[-1][data[-1]['Confirmed']>0].filter(regex=("Country.*")).values)

num_infected_countries = [len(np.unique(data[day][data[day]['Confirmed']>0].filter(regex=("Country.*")).values)) for day in x_dat]

all_countries = dict()
all_countries_values = list()
features = ['Confirmed', 'Deaths', 'Recovered']
all_countries['Confirmed'] = {}
all_countries['Deaths'] = {}
all_countries['Recovered'] = {}

# hard code for Korea
for feature in features:    
    for day in x_dat:
        # Korea, South has two different naming
        tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value='Korea')[feature].values        

        if tmp.size>0:
            all_countries_values.append(sum(tmp))
        else:
            # fill zero for the NaN value in data after computation of fillna
            all_countries_values.append(0)
    all_countries[feature]['Korea, South'] = all_countries_values

    # reset back to initial status
    all_countries_values = []

infected_countries_latest_without_Korea =  np.delete(infected_countries_latest, np.where(infected_countries_latest == 'Korea, South'))
infected_countries_latest_without_ship =  np.delete(infected_countries_latest, np.where(np.logical_or(infected_countries_latest == 'Diamond Princess', infected_countries_latest == 'MS Zaandam')))

for feature in features:
    # Total Confirmed in all countries 
    for country in infected_countries_latest_without_Korea:
        for day in x_dat:
            # tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value=country)[feature].values            
            tmp = StartML.searchByValueColumn(data[day], try_keys=['Country_Region', 'Country/Region'], column=feature, value=country)[feature].values

            if tmp.size>0:
                all_countries_values.append(sum(tmp))
            else:
                # fill zero for the NaN value in data after computation of fillna
                all_countries_values.append(0)
        all_countries[feature][country] = all_countries_values
        
        # reset back to initial status
        all_countries_values = []

# create dictionary of country's population
country_pop_dict = dict()
for country in infected_countries_latest:
    if country in countries:
        country_pop_dict[country]=sdata[sdata['Country_Region']==country]['Population'].values[0]  
    else:
        print("No Information about the population of country ", country)
        country_pop_dict[country]='NaN'

# EXAMPLES last day increasing deaths in US: sum(all_countries['Deaths']['US'][-1]) - sum(all_countries['Deaths']['US'][-2])

'''
HARD-CODE: (SHOULD MOVE TO COVID_IMPORT)
Reason: US Recoveries have been nullified in the same data source, but distributed in other data source.
Need to hard-code to merge partly data together.
See: https://github.com/CSSEGISandData/COVID-19/issues/3464
'''

# re-calculate totalrecovered_by_day_in_US and sum it up with totalrecovered_by_day_without_us (from second data-set USA)
# reinitial the number to zero to clean out all previous values
keep_values_day = len(data) - len(data_us)

for i in np.arange(keep_values_day, len(data), 1):
    # filter column name 'Country_Region' and 'Country/Region'
    colname = data[i].filter(regex=("Country.*")).columns    
    data[i].loc[data[i][colname[0]]=='US', 'Recovered']=0

# Total all recovered cases in all countries changed by day
totalrecovered_by_day_without_us = [sum(data[day]['Recovered']) for day in x_dat]
totalrecovered_by_day_us = [sum(data_us[day]['Recovered'].fillna(0)) for day in x_dat_us]

# Init a list of all zero values
update_part = [0 for i in range(len(totalrecovered_by_day_us))]

j = 0

# get Sum of total values without US and values with US
for i in np.arange(len(data)-len(data_us), len(data), 1):    
    update_part[j] = totalrecovered_by_day_without_us[i] + totalrecovered_by_day_us[j]    
    j = j+1

''' 
All countries CONFIRMED_cases until last day
'''
# Confirmed by Day in every country
y_dat_confirmed_ByDay = dict()

for country in infected_countries_latest:
    # add data of first day with data from day 2 = total present day - total yesterday
    tmp = [(0, all_countries['Confirmed'][country][0])] + [(day+1, all_countries['Confirmed'][country][day+1] - all_countries['Confirmed'][country][day]) for day in x_dat[:-1]]    
    y_dat_confirmed_ByDay.update([(country, tmp)])

# Total all confirmed cases in all countries changed by day 
totalconfirmed_by_day = [sum(data[day]['Confirmed']) for day in x_dat]

# New Increasing/ changes cases in all countries changed by day
newCasesByDay = [totalconfirmed_by_day[0]]+[totalconfirmed_by_day[day+1]-totalconfirmed_by_day[day] for day in x_dat[:-1]]

'''
All Countries FATALITIES_cases until last day
'''
# Death by Day in every country
y_dat_deaths_ByDay = dict()

for country in infected_countries_latest:
    # add data of first day with data from day 2 = total present day - total yesterday
    tmp = [(0, all_countries['Deaths'][country][0])] + [(day+1, all_countries['Deaths'][country][day+1] - all_countries['Deaths'][country][day]) for day in x_dat[:-1]]    
    y_dat_deaths_ByDay.update([(country, tmp)])
    
# Total all fatalities cases in all countries changed by day
totalfatalities_by_day = [sum(data[day]['Deaths']) for day in x_dat]

# New Increasing/ changes Fatalities in ALL COUNTRIES changed by day
newFatalitiesByDay = [totalfatalities_by_day[0]] + [totalfatalities_by_day[day+1] - totalfatalities_by_day[day] for day in x_dat[:-1]]

'''
All Countries RECOVERED_cases until last day
'''

# HARD-CODE for country US
tmp = [sum(StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='US')['Recovered'].values) for i in np.arange(0, keep_values_day, 1)]
all_countries['Recovered']['US'] = tmp + totalrecovered_by_day_us

# Recovered by Day in every country
y_dat_recovered_ByDay = dict()

for country in infected_countries_latest:
    # add data of first day with data from day 2 = total present day - total yesterday
    tmp = [(0,all_countries['Recovered'][country][0])] + [(day+1, all_countries['Recovered'][country][day+1] - all_countries['Recovered'][country][day]) for day in x_dat[:-1]]    
    y_dat_recovered_ByDay.update([(country, tmp)])
    
# Total all recovered cases in all countries changed by day
totalrecovered_by_day = totalrecovered_by_day_without_us[0:keep_values_day] + update_part

# New Increasing/ changes Recovered in ALL COUNTRIES changed by day
newRecoveredByDay = [totalrecovered_by_day[0]] + [totalrecovered_by_day[day+1] - totalrecovered_by_day[day] for day in x_dat[:-1]]

'''
Collect all Information into 3x Data Frames (Confirmed, Fatalities, Recovered)
'''
countries_confirmed = pd.DataFrame.from_dict(data=all_countries['Confirmed'])
countries_fatalities = pd.DataFrame.from_dict(data=all_countries['Deaths'])
countries_recovered = pd.DataFrame.from_dict(data=all_countries['Recovered'])
