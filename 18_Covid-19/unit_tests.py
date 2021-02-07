import os.path as path
import sys
from inspect import getsourcefile

# setup absolute path
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from covid_import import *
# from covid import *

# test data covid_import
def test_total_confirmed_import():
    assert len(total_confirmed) > 0

def test_total_fatalities_import():
    assert len(total_deaths) > 0

def test_total_recovered_withoutUS_import():
    assert len(total_recovered_without_US) > 0

def test_data_length():
    assert len(data) > 0
    
def test_data_us_length():
    assert len(data) > 0    

def test_data_nonan():
    for day in range(len(data)):
        for col in data[day].columns:
            assert not data[day][col].isnull().values.any()

def test_data_us_nonan():
    for day in range(len(data_us)):
        for col in data_us[day].columns:
            assert not data_us[day][col].isnull().values.any()

def test_days_length():
    assert len(x_dat) > 0 and len(x_dat_us) > 0

# TODO: test data covid

# totalconfirmed_by_day > totalrecovered_by_day, totalconfirmed_by_day > totalfatalities_by_day

# np.array(ratioFatalByDay) < np.array(ratioRecByDay)

# us recovered cases

# tbd.