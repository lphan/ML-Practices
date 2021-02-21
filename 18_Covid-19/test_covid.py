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

def test_num_infected_countries_earliest():
    assert len(infected_countries_earliest) > 0

def test_num_infected_countries_latest():
    assert len(infected_countries_latest) > 0

def test_num_infected_countries_equal():
    assert num_infected_countries[-1] == len(infected_countries_latest)

def test_infected_countries_latest():
    for country in infected_countries_latest:
        assert country in all_countries['Confirmed'].keys()

    for country in infected_countries_latest:
        assert country in all_countries['Deaths'].keys()

    for country in infected_countries_latest:
        assert country in all_countries['Recovered'].keys()

# test number confirmed, deaths, recovered of infected countries
# there are two countries with no case confirmed 
def test_infected_countries_confirmed_latest():
    for country in infected_countries_latest:
        assert all_countries['Confirmed'][country][-1] >= 0

def test_infected_countries_deaths_latest():
    for country in infected_countries_latest:
        assert all_countries['Deaths'][country][-1] >= 0

def test_infected_countries_recovered_latest():
    for country in infected_countries_latest:
        assert all_countries['Recovered'][country][-1] >= 0

def test_length_countries_latest():
    assert len(infected_countries_latest) == len(list(all_countries['Confirmed'].keys()))

# test country population dictionary
def test_countries_confirmed_population():
    for country in infected_countries_latest_without_ship:
        assert country_pop_dict[country] > 0

# test total confirmed/ fatalities/ recovered dict in all countries
def test_total_confirmed_dict():
    assert len(total_confirmed) > 0

def test_total_fatalities_dict():
    assert len(total_deaths) > 0

def test_total_recovered_dict():
    assert len(total_recovered_without_US) > 0

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