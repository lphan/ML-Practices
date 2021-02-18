from covid import *

''' 
Top 10 countries with highest/ lowest cases (new cases confirmed, fatality, recovered) last day 
'''
# TODO: UPDATE (change from list to dataframe - rewrite data structure to convert from tuple-list to data frame)

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
# TODO: UPDATE

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
# TODO: UPDATE

# Ratio of Total Confirmed/ Population (certainly >0)
y_dat_ratioConfPop = dict()
for country in all_countries['Confirmed'].keys():
    y_dat_ratioConfPop[country] = np.round((all_countries['Confirmed'][country][-1]/np.double(country_pop_dict[country]))*100, 3)

# Ratio of Total Deaths/ Total Confirmed
y_dat_ratioDeathConf = dict()
for country in all_countries['Confirmed'].keys():
    if all_countries['Confirmed'][country][-1] == 0:
        y_dat_ratioDeathConf[country] = 0
    else: 
        y_dat_ratioDeathConf[country] = np.round((all_countries['Deaths'][country][-1]/all_countries['Confirmed'][country][-1])*100, 3)

# Ratio of Total Deaths/ Population (certainly >0)
y_dat_ratioDeathPop = dict()
for country in all_countries['Deaths'].keys():
    y_dat_ratioDeathPop[country] = np.round((all_countries['Deaths'][country][-1]/np.double(country_pop_dict[country]))*100, 3)

# Ratio of Total Recovered/ Total Confirmed 
y_dat_ratioRecConf = dict()
for country in all_countries['Confirmed'].keys():
    if all_countries['Confirmed'][country][-1] == 0:
        y_dat_ratioRecConf[country] = 0
    else: 
        y_dat_ratioRecConf[country] = np.round((all_countries['Recovered'][country][-1]/all_countries['Confirmed'][country][-1])*100, 3)

# Ratio of Total Recovered/ Population (certainly >0)
y_dat_ratioRecPop = dict()
for country in all_countries['Deaths'].keys():
    y_dat_ratioRecPop[country] = np.round((all_countries['Recovered'][country][-1]/np.double(country_pop_dict[country]))*100, 3)

# Ratio Total_Recovered over Total_Confirmed changed by Day
ratioRecByDay = [np.round(totalrecovered_by_day[day]/totalconfirmed_by_day[day]*100, 3) for day in x_dat]

# Ratio Total_Fatalities over Total_Confirmed changed by Day
ratioFatalByDay = [np.round(totalfatalities_by_day[day]/totalconfirmed_by_day[day]*100, 3) for day in x_dat]

# get basic data (confirmed, fatalities, recovered) into list of dictionary by day
dataconfirmed = [dict() for i in range(len(data))]
datafatal = [dict() for i in range(len(data))]
datarecovered = [dict() for i in range(len(data))]

for i in range(len(data)):
    # print("Day: ", i)
    col = data[i].filter(like='Country').columns[0]
    
    for country in infected_countries_latest:
        dataconfirmed[i][country] = all_countries['Confirmed'][country][i]
        datafatal[i][country] = all_countries['Deaths'][country][i]
        datarecovered[i][country] = all_countries['Recovered'][country][i]

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