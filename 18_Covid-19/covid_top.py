from covid import *

''' 
Top 10 countries with highest/ lowest cases (new cases confirmed, fatality, recovered) last day 
'''
# Example:
# list_confirmed = [(country, y_dat_confirmed_ByDay[country][-1][1]) for country in infected_countries_latest]    
# countries_highestConfByDay = sorted(list_confirmed, key=lambda x: x[1], reverse=True)
# countries_lowestConfByDay = sorted(list_confirmed, key=lambda x: x[1], reverse=False)

countries_ConfLastDay = countries_confirmed.iloc[-2:].diff().iloc[-1]
countries_highestConfByDay = countries_ConfLastDay.sort_values(ascending=False)
countries_lowestConfByDay = countries_ConfLastDay.sort_values(ascending=True)

countries_FatalLastDay = countries_fatalities.iloc[-2:].diff().iloc[-1]
countries_highestFatalByDay = countries_FatalLastDay.sort_values(ascending=False)
countries_lowestFatalByDay = countries_FatalLastDay.sort_values(ascending=True)

countries_RecLastDay = countries_recovered.iloc[-2:].diff().iloc[-1]
countries_highestRecByDay = countries_RecLastDay.sort_values(ascending=False)
countries_lowestRecByDay = countries_RecLastDay.sort_values(ascending=True)

''' 
Top 10 Countries with highest ratio (cases on population) last DAY (see: file UID_ISO_FIPS_LookUp_Table.csv) 
'''
def getTopConfLastDay(topCountries, country_population):
    topCasesPopulation = [((country, value/country_population[country], country_population[country])) for country, value in topCountries.items()]
    topCasesRatioPop = [(tcp[0], tcp[1]) for tcp in topCasesPopulation]
    topCasesPop = [(tcp[0], tcp[2]) for tcp in topCasesPopulation]
    return topCasesPopulation, topCasesRatioPop, topCasesPop

# Ratio of Confirmed (last day)/ Population (Take the first 10 countries from the list)
topConfPopulation, topConfLastDayRatioPop, topConfCountryPop = getTopConfLastDay(topCountries=countries_highestConfByDay.head(10), country_population=country_pop_dict)

# Ratio of Deaths (last day)/ Population (Take the first 10 countries from the list)
topFatalPopulation, topFatalLastDayRatioPop, topFatalCountryPop = getTopConfLastDay(topCountries=countries_highestFatalByDay.head(10), country_population=country_pop_dict)

# Ratio of Recovered (last day)/ Population (Take the first 10 countries from the list)
topRecPopulation, topRecLastDayRatioPop, topRecCountryPop = getTopConfLastDay(topCountries=countries_highestRecByDay.head(10), country_population=country_pop_dict)

''' 
The different Ratio of the Top 10 countries with highest cases 
'''
# TODO: UPDATE - replace with DataFrame (see: below top10 below)

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

# Total Sum of Confirmed, Fatal, Recovered
totalConfirmed = countries_confirmed.tail(1).values.sum()
totalFatal = countries_fatalities.tail(1).values.sum()
totalRecovered = countries_recovered.tail(1).values.sum()

'''
Top 10 highest
'''
top10confirmed = countries_confirmed.tail(1).transpose().sort_values(by=[lastday], ascending=False).rename(columns={lastday: "Confirmed"})
top10confirmed['population']= [country_pop_dict[country] for country in top10confirmed.index]
top10confirmed['RatioConfirmedByPopulation_in_%']= np.round(top10confirmed['Confirmed']/top10confirmed['population'] *100, 4)                                               
top10confirmed['RatioByTotal_in_%']= np.round(top10confirmed['Confirmed']/totalConfirmed*100, 4)

top10fatal = countries_fatalities.tail(1).transpose().sort_values(by=[lastday], ascending=False).rename(columns={lastday: "Fatal"})
top10fatal['population'] = [country_pop_dict[country] for country in top10fatal.index]
top10fatal['RatioFatalByPopulation_in_%'] = np.round(top10fatal['Fatal']/top10fatal['population'] *100, 4)
top10fatal['RatioByTotal_in_%'] = np.round(top10fatal['Fatal']/totalFatal*100, 4)

top10recovered = countries_recovered.tail(1).transpose().sort_values(by=[lastday], ascending=False).rename(columns={lastday: "Recovered"})
top10recovered['population'] = [country_pop_dict[country] for country in top10recovered.index]
top10recovered['RatioRecoveredByPopulation_in_%'] = np.round(top10recovered['Recovered']/top10recovered['population'] *100, 4)
top10recovered['RatioByTotal_in_%'] = np.round(top10recovered['Recovered']/totalRecovered*100, 4)

'''
Top 10 lowest
'''
top10confirmed_lowest = countries_confirmed.tail(1).transpose().sort_values(by=[lastday], ascending=True).rename(columns={lastday: "Confirmed"})
top10confirmed_lowest['population'] = [country_pop_dict[country] for country in top10confirmed_lowest.index]
top10confirmed_lowest['RatioConfirmedByPopulation_in_%'] = np.round(top10confirmed_lowest['Confirmed']/top10confirmed_lowest['population'] *100, 4)
top10confirmed_lowest['RatioByTotal_in_%'] = np.round(top10confirmed_lowest['Confirmed']/totalConfirmed*100, 4)

top10fatal_lowest = countries_fatalities.tail(1).transpose().sort_values(by=[lastday], ascending=True).rename(columns={lastday: "Fatal"})
top10fatal_lowest['population'] = [country_pop_dict[country] for country in top10fatal_lowest.index]
top10fatal_lowest['RatioFatalByPopulation_in_%']= np.round(top10fatal_lowest['Fatal']/top10fatal_lowest['population'] *100, 4)
top10fatal_lowest['RatioByTotal_in_%']= np.round(top10fatal_lowest['Fatal']/totalFatal*100, 4)

top10recovered_lowest = countries_recovered.tail(1).transpose().sort_values(by=[lastday], ascending=True).rename(columns={lastday: "Recovered"})
top10recovered_lowest['population']= [country_pop_dict[country] for country in top10recovered_lowest.index]
top10recovered_lowest['RatioRecoveredByPopulation_in_%']=np.round(top10recovered_lowest['Recovered']/top10recovered_lowest['population'] *100, 4)
top10recovered_lowest['RatioByTotal_in_%']=np.round(top10recovered_lowest['Recovered']/totalRecovered*100, 4)
