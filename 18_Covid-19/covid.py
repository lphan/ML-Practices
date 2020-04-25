# TODO: ANSWERING AD-HOC QUESTIONS
# REFACTORING the CODE

# setup absolute path to location of package Starts and config-file 
from inspect import getsourcefile
import os.path as path, sys

current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from Starts.start import *
from Starts.startml import *
from Starts.startvis import *
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 6

'''
Data Preprocessing 
'''
# Pre-Processing: fill all NaN with 0
data = [data[i].fillna(0) for i in range(len(data))]

x_dat = np.arange(len(data))

# CHINA: Pre-Processing NaN value confirmed_cases
y_dat_cn = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='China')['Confirmed'].values
    for i in range(len(data))]

y_dat_cn = [0 if y.size == 0 else sum(y) for y in y_dat_cn]

# GERMANY: Pre-Processing empty value confirmed_cases
y_dat_de = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Germany')['Confirmed'].values
    for i in range(len(data))]

y_dat_de = [0 if y.size == 0 else sum(y) for y in y_dat_de]

# ITALY: Pre-Processing empty value confirmed_cases
y_dat_it = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Italy')['Confirmed'].values
    for i in range(len(data))]

y_dat_it = [0 if y.size == 0 else sum(y) for y in y_dat_it]

# Republic of Korea: Pre-Processing confirmed_cases
y_dat_kr = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Confirmed'].values
    for i in range(len(data))]

y_dat_kr = [0 if y.size == 0 else sum(y) for y in y_dat_kr]

# JAPAN: Pre-Processing confirmed_cases
y_dat_jp = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value="Japan")['Confirmed'].values
    for i in range(len(data))]

y_dat_jp = [0 if y.size == 0 else sum(y) for y in y_dat_jp]

# US: Pre-Processing confirmed_cases
y_dat_us = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value="US")['Confirmed'].values
    for i in range(len(data))]

y_dat_us = [0 if y.size == 0 else sum(y) for y in y_dat_us]

# AUSTRALIA: Pre-Processing confirmed_cases
y_dat_au = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value="Australia")['Confirmed'].values
    for i in range(len(data))]

y_dat_au = [0 if y.size == 0 else sum(y) for y in y_dat_au]

''' number of all infected countries changed by day '''
# num_infected_countries = [len(data[i][data[i]['Confirmed']>0]['Country/Region'].unique()) for i in range(len(data))]
# filter column by name and convert Pandas frame to Numpy Array
infected_countries_earliest = np.unique(data[0][data[0]['Confirmed']>0].filter(regex=("Country.*")).values)
infected_countries_latest = np.unique(data[-1][data[-1]['Confirmed']>0].filter(regex=("Country.*")).values)

num_infected_countries = [len(np.unique(data[i][data[i]['Confirmed']>0].filter(regex=("Country.*")).values)) for i in range(len(data))]

# Total all confirmed cases in all countries changed by day
totalconfirmed_by_day = [sum(data[i]['Confirmed']) for i in range(len(data))]

# Total all recovered cases in all countries changed by day
totalrecovered_by_day = [sum(data[i]['Recovered']) for i in range(len(data))]

# New Increasing/ changes cases in all countries changed by day
newCasesByDay = [totalconfirmed_by_day[0]]+[totalconfirmed_by_day[i+1]-totalconfirmed_by_day[i] for i in range(len(totalconfirmed_by_day)-1)]

'''
All Countries Fatalities_cases
'''
y_dat_all_fatal = [sum(data[i][data[i]['Deaths'] > 0]['Deaths'].values) for i in range(len(data))]

# New Increasing/ changes Fatalities in all countries changed by day
newFatalitiesByDay = [y_dat_all_fatal[0]]+[y_dat_all_fatal[i+1]-y_dat_all_fatal[i] for i in range(len(y_dat_all_fatal)-1)]

# CHINA: Fatalities_cases
y_dat_death_cn = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='China')['Deaths'].values
    for i in range(len(data))]

y_dat_death_cn = [0 if y.size == 0 else sum(y) for y in y_dat_death_cn]

# GERMANY: Fatalities_cases
y_dat_death_de = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Germany')['Deaths'].values
    for i in range(len(data))]

y_dat_death_de = [0 if y.size == 0 else sum(y) for y in y_dat_death_de]

# ITALY: Fatalities_cases
y_dat_death_it = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Italy')['Deaths'].values
    for i in range(len(data))]

y_dat_death_it = [0 if y.size == 0 else sum(y) for y in y_dat_death_it]

# Republic of Korea: Fatalities_cases					
y_dat_death_kr = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Deaths'].values
    for i in range(len(data))]

y_dat_death_kr = [0 if y.size == 0 else sum(y) for y in y_dat_death_kr]

# JAPAN: Fatalities_cases
y_dat_death_jp = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Japan')['Deaths'].values
    for i in range(len(data))]

y_dat_death_jp = [0 if y.size == 0 else sum(y) for y in y_dat_death_jp]

# USA: Fatalities_cases
y_dat_death_us = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='US')['Deaths'].values
    for i in range(len(data))]

y_dat_death_us = [0 if y.size == 0 else sum(y) for y in y_dat_death_us]

# AUSTRALIA: Fatalities_cases
y_dat_death_au = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Australia')['Deaths'].values
    for i in range(len(data))]

y_dat_death_au = [0 if y.size == 0 else sum(y) for y in y_dat_death_au]

'''
All Countries RECOVERED
'''
# New Increasing/ changes Fatalities in all countries changed by day
y_dat_all_recovered = [sum(data[i][data[i]['Recovered'] > 0]['Recovered'].values) for i in range(len(data))]
newRecoveredByDay = [y_dat_all_recovered[0]] + [y_dat_all_recovered[i+1]-y_dat_all_recovered[i] 
                                                for i in range(len(y_dat_all_recovered)-1)]

y_dat_recovered_cn = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='China')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_cn = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_cn]

y_dat_recovered_de = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Germany')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_de = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_de]

y_dat_recovered_it = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Italy')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_it = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_it]

y_dat_recovered_kr = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_kr = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_kr]

y_dat_recovered_jp = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Japan')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_jp = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_jp]

y_dat_recovered_us = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='US')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_us = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_us]

y_dat_recovered_au = [
    StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Australia')['Recovered'].values
    for i in range(len(data))]
y_dat_recovered_au = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_au]

'''
Total comparison increasing by day in 
Western_culture (top 10 countries: US Germany Italy Spain France UK Swiss Netherland Austria Belgium) 
and
Estern_culture (top 10/ all countries:  China Korea Japan Malaysia Indonesia Thailand Philippine Singapore Taiwan Vietnam)
'''

eu_10_countries = ['Italy', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria',
                   'Belgium', 'Norway']
asia_10countries = ['China', 'Korea', 'Japan', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Singapore',
                    'Taiwan', 'Vietnam']

# total cases by days in 10 EU countries and 10 ASIA countries
eu_byDay = []
asia_byDay = []

# total fatalities by days in 10 EU countries and 10 ASIA countries
eu_deaths_byDay = []
asia_deaths_byDay = []

# total recovered by days in 10 EU countries and 10 ASIA countries
eu_rec_byDay = []
asia_rec_byDay = []

for i in range(len(data)):
    eu_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values
         for ec in eu_10_countries])
    asia_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values
         for ec in asia_10countries])

    eu_deaths_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Deaths'].values
         for ec in eu_10_countries])
    asia_deaths_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Deaths'].values
         for ec in asia_10countries])

    eu_rec_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Recovered'].values
         for ec in eu_10_countries])
    asia_rec_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Recovered'].values
         for ec in asia_10countries])

# --------------- fill empty data by 0
fill_eu_byday = []
for eu in eu_byDay:
    fill_eu_byday.append([np.array([0]) if e.size == 0 else e for e in eu])

fill_asia_byday = []
for asia in asia_byDay:
    fill_asia_byday.append([np.array([0]) if a.size == 0 else a for a in asia])

# --------------- fill by summing all data by day
fill_eu_byday_temp = []
fill_asia_byday_temp = []

fill_eu_fatal_byday_temp = []
fill_asia_fatal_byday_temp = []

fill_eu_recovered_byday_temp= []
fill_asia_recovered_byday_temp = []

for i in range(len(data)):
    fill_eu_byday_temp.append([sum(fill_eu) for fill_eu in fill_eu_byday[i]])
    fill_asia_byday_temp.append([sum(fill_asia) for fill_asia in fill_asia_byday[i]])

    fill_eu_fatal_byday_temp.append([sum(fill_eu) for fill_eu in eu_deaths_byDay[i]])
    fill_asia_fatal_byday_temp.append([sum(fill_asia) for fill_asia in asia_deaths_byDay[i]])

    fill_eu_recovered_byday_temp.append([sum(fill_eu) for fill_eu in eu_rec_byDay[i]])
    fill_asia_recovered_byday_temp.append([sum(fill_asia) for fill_asia in asia_rec_byDay[i]])

''' --------------- Computation the total cases in EU and ASIA (infected cases, fatalities, recovered) '''
eu_total = []
asia_total = []

eu_deaths_total = []
asia_deaths_total = []

eu_rec_total = []
asia_rec_total = []

for i in range(len(data)):
    eu_total.append(sum(fill_eu_byday_temp[i]))
    asia_total.append(sum(fill_asia_byday_temp[i]))

    eu_deaths_total.append(sum(fill_eu_fatal_byday_temp[i]))
    asia_deaths_total.append(sum(fill_asia_fatal_byday_temp[i]))

    eu_rec_total.append(sum(fill_eu_recovered_byday_temp[i]))
    asia_rec_total.append(sum(fill_asia_recovered_byday_temp[i]))

'''
Total of infected cases, fatalities, recovered in the world changed by week
'''
def numberByWeeks(keys):
    for key in keys:
        weeks = []
        week = 1
        for i in range(6, len(data), 7):
            weeks.append((week, data[i][key].values))        
            week = week + 1

        sums = [(k, int(sum(v))) for k,v in weeks]
        
        # Add the last day of current week to sums
        sums.append((week, sum(data[-1][key].values)))
        
        if key is 'Confirmed':
            confirm = sums
        elif key is 'Deaths':
            deaths = sums
        else:
            recovered = sums
                
    return confirm, deaths, recovered
        
confirmedByWeek, deathsByWeek, recoveredByWeek = numberByWeeks(['Confirmed', 'Deaths', 'Recovered'])

''' Top 10 countries with highest cases (new cases, fatality, recovered) changed by day '''
all_countries_conf = [(country, 
                       int(sum(StartML.searchByValue(data[-1], try_keys=['Country_Region', 'Country/Region'], value=country)['Confirmed'].values))
                       - int(sum(StartML.searchByValue(data[-2], try_keys=['Country_Region', 'Country/Region'], value=country)['Confirmed'].values))) 
                     for country in infected_countries_latest]

all_countries_fatal = [(country, 
                        int(sum(StartML.searchByValue(data[-1], try_keys=['Country_Region', 'Country/Region'], value=country)['Deaths'].values))
                        - int(sum(StartML.searchByValue(data[-2], try_keys=['Country_Region', 'Country/Region'], value=country)['Deaths'].values)))
                     for country in infected_countries_latest]

all_countries_rec = [(country, 
                      int(sum(StartML.searchByValue(data[-1], try_keys=['Country_Region', 'Country/Region'], value=country)['Recovered'].values))
                      - int(sum(StartML.searchByValue(data[-2], try_keys=['Country_Region', 'Country/Region'], value=country)['Recovered'].values)))
                     for country in infected_countries_latest]

countries_highestConfByDay = sorted(all_countries_conf, key=lambda x: x[1], reverse=True)
countries_highestFatalByDay = sorted(all_countries_fatal, key=lambda x: x[1], reverse=True)
countries_highestRecByDay = sorted(all_countries_rec, key=lambda x: x[1], reverse=True)

topConf=countries_highestConfByDay[0:10]
topFatal=countries_highestFatalByDay[0:10]
topRec=countries_highestRecByDay[0:10]

''' Top 10 countries with lowest cases (new cases, fatality, recovered) changed by day '''
countries_lowestConfByDay = sorted(all_countries_conf, key=lambda x: x[1], reverse=False)
countries_lowestFatalByDay = sorted(all_countries_fatal, key=lambda x: x[1], reverse=False)
countries_lowestRecByDay = sorted(all_countries_rec, key=lambda x: x[1], reverse=False)

''' Top 10 Countries with highest ratio (cases on population) (see: file UID_ISO_FIPS_LookUp_Table.csv) '''
country_pop = [(country, sdata[sdata['Country_Region']==country]['Population'].values[0]) for country in sdata['Country_Region'].unique()]

topConfPopulation = []
for c in topConf:
    for country in country_pop:
        if country[0] == c[0]:
            topConfPopulation.append((country[0], c[1]/int(country[1]), int(country[1])))
topConfRatioPop = [(tcp[0], tcp[1]) for tcp in topConfPopulation]
topConfPop = [(tcp[0], tcp[2]) for tcp in topConfPopulation]

topFatalPopulation = []
for c in topFatal:
    for country in country_pop:
        if country[0] == c[0]:
            topFatalPopulation.append((country[0], c[1]/int(country[1]), int(country[1])))
topFatalRatioPop = [(tfp[0], tfp[1]) for tfp in topFatalPopulation]
topFatalPop = [(tfp[0], tfp[2]) for tfp in topFatalPopulation]
            
topRecPopulation = []
for c in topRec:
    for country in country_pop:
        if country[0] == c[0]:
            topRecPopulation.append((country[0], c[1]/int(country[1]), int(country[1])))
topRecRatioPop = [(trp[0], trp[1]) for trp in topRecPopulation]
topRecPop = [(trp[0], trp[2]) for trp in topRecPopulation]