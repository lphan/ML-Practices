import os.path as path
import sys
from inspect import getsourcefile

# setup absolute path
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from covid import *

# test number of infected countries 
def test_num_infected_countries():
    assert len(num_infected_countries) > 0

def test_num_confirmed_countries():
    assert len(countries_confirmed) > 0

def test_num_fatal_countries_latest():
    assert len(countries_fatalities) > 0

def test_num_recovered_countries_latest():
    assert len(countries_recovered) > 0

# test number confirmed, deaths, recovered of infected countries
# there are two countries with no case confirmed 
def test_infected_countries_confirmed_latest():
    for country in countries_confirmed.keys():
        assert countries_confirmed[country][-1] >= 0

def test_infected_countries_deaths_latest():
    for country in countries_fatalities.keys():
        assert countries_fatalities[country][-1] >= 0

def test_infected_countries_recovered_latest():
    for country in countries_recovered.keys():
        assert countries_recovered[country][-1] >= 0

def test_length_countries_latest():
    assert totalconfirmed_by_day == len(list(countries_confirmed.columns))

# issue bug-test in function searchByValue (contain-function) has value = 0 but searchByValueColumn and groupby function return value > 0
features = ['Confirmed', 'Deaths', 'Recovered']

def test_data_searchByValue():
    for feature in features:   
        for country in ['Congo (Brazzaville)', 'Congo (Kinshasa)']:            
            assert sum(StartML.searchByValue(data[-1], try_keys=['Country_Region', 'Country/Region'], value=country)[feature].values) == 0

def test_data_total_confirmed_byday():
    for day in x_dat:
        assert totalconfirmed_by_day[day] == sum(data[day]['Confirmed']) 

def test_data_total_fatal_byday():
    for day in x_dat:
        assert totalfatalities_by_day[day] == sum(data[day]['Deaths']) 

def test_data_total_rec_byday():
    for day in x_dat:
        assert totalrecovered_by_day[day] == sum(data[day]['Recovered']) 

# def test_data_nonan():
#     for day in range(len(data)):
#         for col in data[day].columns:
#             assert not data[day][col].isnull().values.any()

# def test_data_us_nonan():
#     for day in range(len(data_us)):
#         for col in data_us[day].columns:
#             assert not data_us[day][col].isnull().values.any()

# def test_days_length():
#     assert len(x_dat) > 0 and len(x_dat_us) > 0