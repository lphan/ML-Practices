# ANSWERING the AD-HOC QUESTIONS
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
y_dat_confirmed = dict()
for country in all_countries['Confirmed'].keys():
    y_dat_confirmed[country] = [sum(all_countries['Confirmed'][country][day]) for day in x_dat]

# Death by Day in every country
y_dat_confirmed_ByDay = dict()

for country in infected_countries_latest:
    # add data of first day with data from day 2 = total present day - total yesterday
    tmp = [(0, y_dat_confirmed[country][0])] + [(day+1, y_dat_confirmed[country][day+1] - y_dat_confirmed[country][day]) for day in x_dat[:-1]]    
    y_dat_confirmed_ByDay.update([(country, tmp)])

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
# first_group = ['Italy', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria', 'Belgium', 'Luxembourg']
first_group = ['Norway', 'Sweden', 'Denmark', 'Finland']
second_group = ['China', 'Korea, South', 'Japan', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Singapore', 'Taiwan*', 'Vietnam']

# first_group: female, second_group: male
# first_group = ['Germany', 'Taiwan*', 'New Zealand', 'Iceland', 'Finland', 'Norway', 'Denmark']
# second_group = ['US', 'Brazil', 'United Kingdom', 'Russia', 'Italy', 'Spain', 'France']

first_group_population = [country_pop_dict[country] for country in first_group]
second_group_population = [country_pop_dict[country] for country in second_group]

# total cases by days in 10 EU countries and 10 ASIA countries
firstgroup_byDay = []
secondgroup_byDay = []

# total fatalities by days in 10 EU countries and 10 ASIA countries
firstgroup_deaths_byDay = []
secondgroup_deaths_byDay = []

# total recovered by days in 10 EU countries and 10 ASIA countries
firstgroup_rec_byDay = []
secondgroup_rec_byDay = []

# iterate all days
for i in x_dat:
    firstgroup_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values
         for ec in first_group])
    secondgroup_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Confirmed'].values
         for ac in second_group])

    firstgroup_deaths_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Deaths'].values
         for ec in first_group])
    secondgroup_deaths_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Deaths'].values
         for ac in second_group])

    firstgroup_rec_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Recovered'].values
         for ec in first_group])
    secondgroup_rec_byDay.append(
        [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Recovered'].values
         for ac in second_group])

# fill empty data by 0
# @jit(nopython=True)
# @njit
def getfill(fillbyday):
    return [[np.array([0]) if fi.size == 0 else fi for fi in fill] for fill in fillbyday]
    
fill_firstgroup_byDay = getfill(fillbyday=firstgroup_byDay)
fill_secondgroup_byDay = getfill(fillbyday=secondgroup_byDay)

fill_firstgroup_byDay_temp = []
fill_secondgroup_byDay_temp = []

fill_firstgroup_fatal_byday_temp = []
fill_secondgroup_fatal_byday_temp = []

fill_firstgroup_recovered_byday_temp= []
fill_secondgroup_recovered_byday_temp = []

# iterate all days
for i in x_dat:
    fill_firstgroup_byDay_temp.append([sum(fill_firstgroup) for fill_firstgroup in fill_firstgroup_byDay[i]])
    fill_secondgroup_byDay_temp.append([sum(fill_secondgroup) for fill_secondgroup in fill_secondgroup_byDay[i]])

    fill_firstgroup_fatal_byday_temp.append([sum(fill_firstgroup) for fill_firstgroup in firstgroup_deaths_byDay[i]])
    fill_secondgroup_fatal_byday_temp.append([sum(fill_secondgroup) for fill_secondgroup in secondgroup_deaths_byDay[i]])

    fill_firstgroup_recovered_byday_temp.append([sum(fill_firstgroup) for fill_firstgroup in firstgroup_rec_byDay[i]])
    fill_secondgroup_recovered_byday_temp.append([sum(fill_secondgroup) for fill_secondgroup in secondgroup_rec_byDay[i]])


# Computation the total cases in EU and ASIA (infected cases, fatalities, recovered) 
firstgroup_total = []
secondgroup_total = []

firstgroup_deaths_total = []
secondgroup_deaths_total = []

firstgroup_rec_total = []
secondgroup_rec_total = []

for i in x_dat:
    firstgroup_total.append(sum(fill_firstgroup_byDay_temp[i]))
    secondgroup_total.append(sum(fill_secondgroup_byDay_temp[i]))

    firstgroup_deaths_total.append(sum(fill_firstgroup_fatal_byday_temp[i]))
    secondgroup_deaths_total.append(sum(fill_secondgroup_fatal_byday_temp[i]))

    firstgroup_rec_total.append(sum(fill_firstgroup_recovered_byday_temp[i]))
    secondgroup_rec_total.append(sum(fill_secondgroup_recovered_byday_temp[i]))

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
Top 10 countries with highest/ lowest cases (new cases confirmed, fatality, recovered) last day 
'''
list_confirmed = [(country, y_dat_confirmed_ByDay[country][-1][1]) for country in infected_countries_latest]    
countries_highestConfByDay = sorted(list_confirmed, key=lambda x: x[1], reverse=True)
countries_lowestConfByDay = sorted(list_confirmed, key=lambda x: x[1], reverse=False)

list_fatal = [(country, y_dat_deaths_ByDay[country][-1][1]) for country in infected_countries_latest]
countries_highestFatalByDay = sorted(list_fatal, key=lambda x: x[1], reverse=True)
countries_lowestFatalByDay = sorted(list_fatal, key=lambda x: x[1], reverse=False)

list_recovered = [(country, y_dat_recovered_ByDay[country][-1][1]) for country in infected_countries_latest]
countries_highestRecByDay = sorted(list_recovered, key=lambda x: x[1], reverse=True)
countries_lowestRecByDay = sorted(list_recovered, key=lambda x: x[1], reverse=False)

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


''' Number of all infected countries changed by day '''
# filter column by name and convert Pandas frame to Numpy Array
num_infected_countries = [len(np.unique(data[day][data[day]['Confirmed']>0].filter(regex=("Country.*")).values)) for day in x_dat]
# num_infected_countries = [len(np.unique(data[i][data[i]['Confirmed']>0].filter(regex=("Country.*")).values)) for i in range(len(data))]

# get basic data (confirmed, fatalities, recovered) into list of dictionary by day
dataconfirmed = [dict() for i in range(len(data))]
datafatal = [dict() for i in range(len(data))]
datarecovered = [dict() for i in range(len(data))]

for i in range(len(data)):
    # print("Day: ", i)
    col = data[i].filter(like='Country').columns[0]
    
    for country in infected_countries_latest:
        dataconfirmed[i][country] = y_dat_confirmed[country][i]
        datafatal[i][country] = y_dat_deaths[country][i]
        datarecovered[i][country] = y_dat_recovered[country][i]

# create 3x data frames (columns = all countries, rows = all days) for confirmed, deaths, and recovered
pdConfirmed = pd.DataFrame(data=dataconfirmed, columns=infected_countries_latest)
pdDeaths = pd.DataFrame(data=datafatal, columns=infected_countries_latest)
pdRecovered = pd.DataFrame(data=datarecovered, columns=infected_countries_latest)

# Select top and bottom values
totalConfirmed = pdConfirmed.tail(1).values.sum()
totalFatal = pdDeaths.tail(1).values.sum()
totalRecovered = pdRecovered.tail(1).values.sum()

# Top 10 highest
top10confirmed = pdConfirmed.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
top10confirmed['RatioByTotal_in_%']=[np.round(top10confirmed.loc[country].values[0]/totalConfirmed*100, 4) if totalConfirmed>0 else 0 for country in top10confirmed.index]

top10fatal = pdDeaths.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
top10fatal['RatioByTotal_in_%']=[np.round(top10fatal.loc[country].values[0]/totalFatal*100, 4) if totalFatal>0 else 0 for country in top10fatal.index]

top10recovered = pdRecovered.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
top10recovered['RatioByTotal_in_%']=[np.round(top10recovered.loc[country].values[0]/totalRecovered*100, 4) if totalRecovered>0 else 0 for country in top10recovered.index]

# Top 10 lowest
top10confirmed_lowest = pdConfirmed.tail(1).transpose().sort_values(by=[lastday], ascending=True)
top10confirmed_lowest['RatioConfirmedByPopulation_in_%']=[np.round(top10confirmed_lowest.loc[country].values[0]/sdata[sdata['Country_Region']==country]['Population'].values[0] *100, 4) if sdata[sdata['Country_Region']==country]['Population'].values[0]>0 else 0 for country in top10confirmed_lowest.index]
top10confirmed_lowest['population']= [sdata[sdata['Country_Region']==country]['Population'].values[0] for country in top10confirmed_lowest.index]

top10fatal_lowest = pdDeaths.tail(1).transpose().sort_values(by=[lastday], ascending=True)
top10fatal_lowest['RatioFatalByConfirmed_in_%']=[np.round(top10fatal_lowest.loc[country].values[0]/ pdConfirmed[country].tail(1).values[0] *100, 4) if pdConfirmed[country].tail(1).values[0]>0 else 0 for country in top10fatal_lowest.index]
top10fatal_lowest['Confirmed']= [pdConfirmed[country].tail(1).values[0] for country in top10fatal_lowest.index]

top10recovered_lowest = pdRecovered.tail(1).transpose().sort_values(by=[lastday], ascending=True)
top10recovered_lowest['RatioRecoveredByConfirmed_in_%']=[np.round(top10recovered_lowest.loc[country].values[0]/ pdConfirmed[country].tail(1).values[0] *100, 4) if pdConfirmed[country].tail(1).values[0]>0 else 0 for country in top10recovered_lowest.index]
top10recovered_lowest['Confirmed']= [pdConfirmed[country].tail(1).values[0] for country in top10recovered_lowest.index]