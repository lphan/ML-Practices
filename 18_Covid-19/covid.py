# setup absolute path to location of package Starts and config-file 
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
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

# number of all infected countries changed by day
# num_infected_countries = [len(data[i][data[i]['Confirmed']>0]['Country/Region'].unique()) for i in range(len(data))]
# filter column by name and convert Pandas frame to Numpy Array
num_infected_countries = [len(np.unique(data[i][data[i]['Confirmed']>0].filter(regex=("Country.*")).values)) for i in range(len(data))]

# Total all confirmed cases in all countries changed by day
totalconfirmed_by_day = [sum(data[i]['Confirmed']) for i in range(len(data))] 

# Total all recovered cases in all countries changed by day
totalrecovered_by_day = [sum(data[i]['Recovered']) for i in range(len(data))]

# CHINA: Pre-Processing NaN value confirmed_cases
y_dat_cn = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='China')['Confirmed'].values 
            for i in range(len(data))]

y_dat_cn = [0 if y.size == 0 else sum(y) for y in y_dat_cn]

# GERMANY: Pre-Processing empty value confirmed_cases
y_dat_de = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Germany')['Confirmed'].values
            for i in range(len(data))]

y_dat_de = [0 if y.size == 0 else sum(y) for y in y_dat_de]

# ITALY: Pre-Processing empty value confirmed_cases
y_dat_it = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Italy')['Confirmed'].values
            for i in range(len(data))]

y_dat_it = [0 if y.size == 0 else sum(y) for y in y_dat_it]

# Republic of Korea: Pre-Processing confirmed_cases
y_dat_kr = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Confirmed'].values
            for i in range(len(data))]

y_dat_kr = [0 if y.size == 0 else sum(y) for y in y_dat_kr]

# JAPAN: Pre-Processing confirmed_cases
y_dat_jp = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value="Japan")['Confirmed'].values
            for i in range(len(data))]

y_dat_jp = [0 if y.size == 0 else sum(y) for y in y_dat_jp]

# US: Pre-Processing confirmed_cases
y_dat_us = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value="US")['Confirmed'].values
            for i in range(len(data))]

y_dat_us = [0 if y.size == 0 else sum(y) for y in y_dat_us]

# AUSTRALIA: Pre-Processing confirmed_cases
y_dat_au = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value="Australia")['Confirmed'].values
            for i in range(len(data))]

y_dat_au = [0 if y.size == 0 else sum(y) for y in y_dat_au]

'''
All Countries Fatalities_cases
'''
y_dat_all_fatal = [sum(data[i][data[i]['Deaths']>0]['Deaths'].values) for i in range(len(data))]

# CHINA: Fatalities_cases
y_dat_death_cn = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='China')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_cn = [0 if y.size == 0 else sum(y) for y in y_dat_death_cn] 

# GERMANY: Fatalities_cases
y_dat_death_de = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Germany')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_de = [0 if y.size == 0 else sum(y) for y in y_dat_death_de]

# ITALY: Fatalities_cases
y_dat_death_it = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Italy')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_it = [0 if y.size == 0 else sum(y) for y in y_dat_death_it]

# Republic of Korea: Fatalities_cases					
y_dat_death_kr = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_kr = [0 if y.size == 0 else sum(y) for y in y_dat_death_kr]

# JAPAN: Fatalities_cases
y_dat_death_jp = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Japan')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_jp = [0 if y.size == 0 else sum(y) for y in y_dat_death_jp]

# USA: Fatalities_cases
y_dat_death_us = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='US')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_us = [0 if y.size == 0 else sum(y) for y in y_dat_death_us]

# AUSTRALIA: Fatalities_cases
y_dat_death_au = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Australia')['Deaths'].values 
					for i in range(len(data))]

y_dat_death_au = [0 if y.size == 0 else sum(y) for y in y_dat_death_au]

'''
All Countries RECOVERED
'''
y_dat_all_recovered = [sum(data[i][data[i]['Recovered']>0]['Recovered'].values) for i in range(len(data))]

y_dat_recovered_cn = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='China')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_cn = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_cn]

y_dat_recovered_de = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Germany')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_de = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_de]

y_dat_recovered_it = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Italy')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_it = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_it]

y_dat_recovered_kr = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Korea')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_kr = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_kr]

y_dat_recovered_jp = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Japan')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_jp = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_jp]

y_dat_recovered_us = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='US')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_us = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_us]

y_dat_recovered_au = [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value='Australia')['Recovered'].values 
					for i in range(len(data))]
y_dat_recovered_au = [0 if y.size == 0 else sum(y) for y in y_dat_recovered_au]

'''
Total comparison by day
Western_culture (top 10 countries: US Germany Italy Spain France UK Swiss Netherland Austria Belgium) 
vs 
Estern_culture (top 10/ all countries:  Chian Korea Japan Malaysia Indonesia Thailand Philippine Singapore Taiwan Vietnam)

Total increasing by day
'''

eu_10_countries = ['Italy', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria', 'Belgium', 'Norway']
asia_10countries = ['China','Korea','Japan','Malaysia','Indonesia','Thailand','Philippines','Singapore','Taiwan','Vietnam']

eu_byDay = []
asia_byDay = []
for i in range(len(data)):
	eu_byDay.append([StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values 
            			for ec in eu_10_countries])
	asia_byDay.append([StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values 
            			for ec in asia_10countries])

# fill empty data by 0
fill_eu_byday = []
for eu in eu_byDay:
    fill_eu_byday.append([np.array([0]) if eub.size == 0 else eub for eub in eu])

fill_asia_byday = []
for asia in asia_byDay:
    fill_asia_byday.append([np.array([0]) if a.size == 0 else a for a in asia])

# fill by summing all data by day
fill_eu_byday_temp = []
for i in range(len(data)):
    fill_eu_byday_temp.append([sum(fill_eu) for fill_eu in fill_eu_byday[i]])
    
fill_asia_byday_temp = []
for i in range(len(data)):
    fill_asia_byday_temp.append([sum(fill_asia) for fill_asia in fill_asia_byday[i]])
    
# Example: Day 1, day 50
# fill_eu_byday_temp[0], fill_eu_byday_temp[50]

eu_total = []
for i in range(len(data)):    
    eu_total.append(sum(fill_eu_byday_temp[i]))
    
asia_total = []
for i in range(len(data)):    
    asia_total.append(sum(fill_asia_byday_temp[i]))