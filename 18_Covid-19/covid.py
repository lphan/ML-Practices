# ANSWERING the AD-HOC QUESTIONS

from covid_import import *

'''
Data Preprocessing 
'''
# search all negative values in data or NaN and fill them all with 0 
data_date = dict([(i, files[i]) for i in range(len(files))]) 

new_days_list = np.arange(0, len(list(data_date.keys())[:-1]), 1)  # re-index from 0
for day in new_days_list:
    data[day][['Confirmed','Deaths','Recovered']] = data[day][['Confirmed','Deaths','Recovered']].mask(data[day][['Confirmed','Deaths','Recovered']]<0, 0)

# collect all data into list all countries of tuple (country, confirmed), (country, fatalities), (country, recovered)
countries = sdata['Country_Region'].unique()

''' Number of all infected countries changed by day '''
# filter column by name and convert Pandas frame to Numpy Array
infected_countries_earliest = np.unique(data[0][data[0]['Confirmed']>0].filter(regex=("Country.*")).values)
infected_countries_latest = np.unique(data[-1][data[-1]['Confirmed']>0].filter(regex=("Country.*")).values)

num_infected_countries = [len(np.unique(data[day][data[day]['Confirmed']>0].filter(regex=("Country.*")).values)) for day in x_dat]

all_countries = dict()
all_countries['Confirmed'] = {}
all_countries['Deaths'] = {}
all_countries['Recovered'] = {}
all_countries_values = list()

# Total Confirmed in all countries
for country in infected_countries_latest:
    for day in x_dat:
        # hard code for Korea
        if country == "Korea, South":
            tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Confirmed'].values
        else:
            tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value=country)['Confirmed'].values            
        if tmp.size>0:
            all_countries_values.append(tmp)
        else:
            # fill zero for the NaN value in data after computation of fillna
            all_countries_values.append(np.array([0]))
    all_countries['Confirmed'][country] = all_countries_values
    
    # reset back to initial status
    all_countries_values = []
    
# Total Deaths in all countries
for country in infected_countries_latest:
    for day in x_dat:
        # hard code for Korea
        if country == "Korea, South":
            tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Deaths'].values
        else:
            tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value=country)['Deaths'].values
        if tmp.size>0:
            all_countries_values.append(tmp)
        else:
            # fill zero for the NaN value in data after computation of fillna
            all_countries_values.append(np.array([0]))
    all_countries['Deaths'][country] = all_countries_values
    
    # reset back to initial status
    all_countries_values = []
    
# Total Recovered in all countries
for country in infected_countries_latest:
    for day in x_dat:
        # hard code for Korea
        if country == "Korea, South":
            tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Recovered'].values
        else:
            tmp = StartML.searchByValue(data[day], try_keys=['Country_Region', 'Country/Region'], value=country)['Recovered'].values
        if tmp.size>0:
            all_countries_values.append(tmp)
        else:
            # fill zero for the NaN value in data after computation of fillna
            all_countries_values.append(np.array([0]))
    all_countries['Recovered'][country] = all_countries_values
    
    # reset back to initial status
    all_countries_values = []

country_pop_dict = dict()
for country in infected_countries_latest:
    if country in countries:
        country_pop_dict[country]=sdata[sdata['Country_Region']==country]['Population'].values[0]  
    else:
        print("No Information about the population of country ", country)
        country_pop_dict[country]='NaN'

# EXAMPLES last day increasing deaths in US: sum(all_countries['Deaths']['US'][-1]) - sum(all_countries['Deaths']['US'][-2])

'''
HARD-CODE:
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
y_dat_confirmed = dict()
for country in all_countries['Confirmed'].keys():
    y_dat_confirmed[country] = [sum(all_countries['Confirmed'][country][day]) for day in x_dat]

# Total all confirmed cases in all countries changed by day
totalconfirmed_by_day = [sum(data[day]['Confirmed']) for day in x_dat]

# New Increasing/ changes cases in all countries changed by day
newCasesByDay = [totalconfirmed_by_day[0]]+[totalconfirmed_by_day[day+1]-totalconfirmed_by_day[day] for day in x_dat[:-1]]

'''
All Countries FATALITIES_cases until last day
'''
y_dat_deaths = dict()
for country in all_countries['Deaths'].keys():
    y_dat_deaths[country] = [sum(all_countries['Deaths'][country][day]) for day in x_dat]

# Death by Day in every country
y_dat_deaths_ByDay = dict()

for country in infected_countries_latest:
    # add data of first day with data from day 2 = total present day - total yesterday
    tmp = [(0, y_dat_deaths[country][0])] + [(day+1, y_dat_deaths[country][day+1] - y_dat_deaths[country][day]) for day in x_dat[:-1]]    
    y_dat_deaths_ByDay.update([(country, tmp)])
    
# Total all fatalities cases in all countries changed by day
totalfatalities_by_day = [sum(data[day]['Deaths']) for day in x_dat]

# New Increasing/ changes Fatalities in ALL COUNTRIES changed by day
newFatalitiesByDay = [totalfatalities_by_day[0]] + [totalfatalities_by_day[day+1] - totalfatalities_by_day[day] for day in x_dat[:-1]]

'''
All Countries RECOVERED_cases until last day
'''
y_dat_recovered = dict()
for country in all_countries['Recovered'].keys():
    y_dat_recovered[country] = [sum(all_countries['Recovered'][country][day]) for day in x_dat]

# HARD-CODE for country US
tmp = [sum(StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='US')['Recovered'].values) for i in np.arange(0, keep_values_day, 1)]
y_dat_recovered['US'] = tmp + totalrecovered_by_day_us

# Recovered by Day in every country
y_dat_recovered_ByDay = dict()

for country in infected_countries_latest:
    # add data of first day with data from day 2 = total present day - total yesterday
    tmp = [(0,y_dat_recovered[country][0])] + [(day+1, y_dat_recovered[country][day+1] - y_dat_recovered[country][day]) for day in x_dat[:-1]]    
    y_dat_recovered_ByDay.update([(country, tmp)])
    
# Total all recovered cases in all countries changed by day
totalrecovered_by_day = totalrecovered_by_day_without_us[0:keep_values_day] + update_part

# New Increasing/ changes Recovered in ALL COUNTRIES changed by day
newRecoveredByDay = [totalrecovered_by_day[0]] + [totalrecovered_by_day[day+1] - totalrecovered_by_day[day] for day in x_dat[:-1]]

'''
Total comparison increasing by day in 
Western_culture (10 countries: US Germany Italy Spain France UK Swiss Netherland Austria Belgium) 
and
Estern_culture (10 countries:  China Korea Japan Malaysia Indonesia Thailand Philippine Singapore Taiwan Vietnam)
'''
eu10_countries = ['Italy', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria',
                   'Belgium', 'Norway']
asia10_countries = ['China', 'Korea, South', 'Japan', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Singapore',
                    'Taiwan*', 'Vietnam']

eu10_population = [country_pop_dict[country] for country in eu10_countries]
asia10_population = [country_pop_dict[country] for country in asia10_countries]

# total cases by days in 10 EU countries and 10 ASIA countries
eu_byDay = []
asia_byDay = []

# total fatalities by days in 10 EU countries and 10 ASIA countries
eu_deaths_byDay = []
asia_deaths_byDay = []

# total recovered by days in 10 EU countries and 10 ASIA countries
eu_rec_byDay = []
asia_rec_byDay = []

# iterate all days
for i in x_dat:
    eu_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values
         for ec in eu10_countries])
    asia_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Confirmed'].values
         for ac in asia10_countries])

    eu_deaths_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Deaths'].values
         for ec in eu10_countries])
    asia_deaths_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Deaths'].values
         for ac in asia10_countries])

    eu_rec_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Recovered'].values
         for ec in eu10_countries])
    asia_rec_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Recovered'].values
         for ac in asia10_countries])

# fill empty data by 0
# @jit(nopython=True)
# @njit
def getfill(fillbyday):
    return [[np.array([0]) if fi.size == 0 else fi for fi in fill] for fill in fillbyday]
    
fill_eu_byday = getfill(fillbyday=eu_byDay)
fill_asia_byday = getfill(fillbyday=asia_byDay)

# fill_eu_byday = []
# for eu in eu_byDay:
#     fill_eu_byday.append([np.array([0]) if e.size == 0 else e for e in eu])

# fill_asia_byday = []
# for asia in asia_byDay:
#     fill_asia_byday.append([np.array([0]) if a.size == 0 else a for a in asia])

# fill by summing all data by day
# Source: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#functions
# @njit
# def getfilleu(): 
#     result = []
#     for i in x_dat:
#         result.append([np.sum(fill_eu) for fill_eu in fill_eu_byday[i]])
#     return result
#
# fill_eu_byday_temp = getfilleu()  

fill_eu_byday_temp = []
fill_asia_byday_temp = []

fill_eu_fatal_byday_temp = []
fill_asia_fatal_byday_temp = []

fill_eu_recovered_byday_temp= []
fill_asia_recovered_byday_temp = []

# iterate all days
for i in x_dat:
    fill_eu_byday_temp.append([sum(fill_eu) for fill_eu in fill_eu_byday[i]])
    fill_asia_byday_temp.append([sum(fill_asia) for fill_asia in fill_asia_byday[i]])

    fill_eu_fatal_byday_temp.append([sum(fill_eu) for fill_eu in eu_deaths_byDay[i]])
    fill_asia_fatal_byday_temp.append([sum(fill_asia) for fill_asia in asia_deaths_byDay[i]])

    fill_eu_recovered_byday_temp.append([sum(fill_eu) for fill_eu in eu_rec_byDay[i]])
    fill_asia_recovered_byday_temp.append([sum(fill_asia) for fill_asia in asia_rec_byDay[i]])

''' 
Computation the total cases in EU and ASIA (infected cases, fatalities, recovered) 
'''
eu_total = []
asia_total = []

eu_deaths_total = []
asia_deaths_total = []

eu_rec_total = []
asia_rec_total = []

for i in x_dat:
    eu_total.append(sum(fill_eu_byday_temp[i]))
    asia_total.append(sum(fill_asia_byday_temp[i]))

    eu_deaths_total.append(sum(fill_eu_fatal_byday_temp[i]))
    asia_deaths_total.append(sum(fill_asia_fatal_byday_temp[i]))

    eu_rec_total.append(sum(fill_eu_recovered_byday_temp[i]))
    asia_rec_total.append(sum(fill_asia_recovered_byday_temp[i]))

''' 
Total of infected cases, fatalities, recovered in the world changed by week 
'''
# @jit(nopython=True)
# @njit
def numberByWeeks(data):
    weeks = list()

    # First week
    weeks.append((1, data[4]))

    # start from second week
    week = 2
    for i in range(11, len(x_dat), 7):
        weeks.append((week, data[i]))
        week = week + 1
    
    # Add the last day of current week to sums
    weeks.append((week, data[-1]))

    return weeks

confirmedByWeek = numberByWeeks(data=totalconfirmed_by_day)
deathsByWeek = numberByWeeks(data=totalfatalities_by_day)
recoveredByWeek = numberByWeeks(data=totalrecovered_by_day)

''' 
Top 10 countries with highest cases (new cases, fatality, recovered) changed by day 
'''
# @jit(nopython=True)
def top10countrieshighest(keyword, infected_countries, reverse=True):    
    all_countries_lastday = [(country, sum(all_countries[keyword][country][-1]) - sum(all_countries[keyword][country][-2])) for country in infected_countries]
    return sorted(all_countries_lastday, key=lambda x: x[1], reverse=True)
    
countries_highestConfByDay = top10countrieshighest(keyword='Confirmed', infected_countries=infected_countries_latest)
countries_highestFatalByDay = top10countrieshighest(keyword='Deaths', infected_countries=infected_countries_latest)
countries_highestRecByDay = top10countrieshighest(keyword='Recovered', infected_countries=infected_countries_latest)

''' 
Top 10 countries with lowest cases (new cases, fatality, recovered) changed by day 
'''
countries_highestConfByDay = top10countrieshighest(keyword='Confirmed', infected_countries=infected_countries_latest, reverse=False)
countries_highestFatalByDay = top10countrieshighest(keyword='Deaths', infected_countries=infected_countries_latest, reverse=False)
countries_highestRecByDay = top10countrieshighest(keyword='Recovered', infected_countries=infected_countries_latest, reverse=False)

''' 
Top 10 Countries with highest ratio (cases on population) last DAY (see: file UID_ISO_FIPS_LookUp_Table.csv) 
'''
# @jit(nopython=True)
def getTopConfLastDay(topCountries, country_population):
    topCasesPopulation = [((country[0], country[1]/int(country_population[country[0]]), int(country_population[country[0]]))) for country in topCountries]
    topCasesRatioPop = [(tcp[0], tcp[1]) for tcp in topCasesPopulation]
    topCasesPop = [(tcp[0], tcp[2]) for tcp in topCasesPopulation]
    return topCasesPopulation, topCasesRatioPop, topCasesPop

# Ratio of Confirmed (last day)/ Population (Take the first 10 countries from the list)
topConfPopulation, topConfLastDayRatioPop, topConfCountryPop = getTopConfLastDay(topCountries=countries_highestConfByDay[0:10], country_population=country_pop_dict)

# Ratio of Deaths (last day)/ Population (Take the first 10 countries from the list)
topFatalPopulation, topFatalLastDayRatioPop, topFatalCountryPop = getTopConfLastDay(topCountries=countries_highestFatalByDay[0:10], country_population=country_pop_dict)

# Ratio of Recovered (last day)/ Population (Take the first 10 countries from the list)
topRecPopulation, topRecLastDayRatioPop, topRecCountryPop = getTopConfLastDay(topCountries=countries_highestRecByDay[0:10], country_population=country_pop_dict)

''' 
The different Ratio of the Top 10 countries with highest cases 
'''
# Ratio of Total Confirmed/ Population (certainly >0)
y_dat_ratioConfPop = dict()
for country in all_countries['Confirmed'].keys():
    y_dat_ratioConfPop[country] = np.round((y_dat_confirmed[country][-1]/np.double(country_pop_dict[country]))*100, 3)

# Ratio of Total Deaths/ Total Confirmed
y_dat_ratioDeathConf = dict()
for country in all_countries['Confirmed'].keys():
    if y_dat_confirmed[country][-1] == 0:
        y_dat_ratioDeathConf[country] = 0
    else: 
        y_dat_ratioDeathConf[country] = np.round((y_dat_deaths[country][-1]/y_dat_confirmed[country][-1])*100, 3)

# Ratio of Total Deaths/ Population (certainly >0)
y_dat_ratioDeathPop = dict()
for country in all_countries['Deaths'].keys():
    y_dat_ratioDeathPop[country] = np.round((y_dat_deaths[country][-1]/np.double(country_pop_dict[country]))*100, 3)

# Ratio of Total Recovered/ Total Confirmed 
y_dat_ratioRecConf = dict()
for country in all_countries['Confirmed'].keys():
    if y_dat_confirmed[country][-1] == 0:
        y_dat_ratioRecConf[country] = 0
    else: 
        y_dat_ratioRecConf[country] = np.round((y_dat_recovered[country][-1]/y_dat_confirmed[country][-1])*100, 3)

# Ratio of Total Recovered/ Population (certainly >0)
y_dat_ratioRecPop = dict()
for country in all_countries['Deaths'].keys():
    y_dat_ratioRecPop[country] = np.round((y_dat_recovered[country][-1]/np.double(country_pop_dict[country]))*100, 3)

# Ratio Total_Recovered over Total_Confirmed changed by Day
ratioRecByDay = [np.round(totalrecovered_by_day[day]/totalconfirmed_by_day[day]*100, 3) for day in x_dat]

# Ratio Total_Fatalities over Total_Confirmed changed by Day
ratioFatalByDay = [np.round(totalfatalities_by_day[day]/totalconfirmed_by_day[day]*100, 3) for day in x_dat]