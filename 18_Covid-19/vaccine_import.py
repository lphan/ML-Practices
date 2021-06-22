# setup absolute path to location of package Starts and config-file
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

# from numba import jit
# from numba import njit
from Starts.start import Start
# from Starts.startml import *
# from Starts.startvis import *
from matplotlib.pylab import rcParams

# Import general data (without US-recovered)
Start._arguments()

# Import global data 
raw_data_us = './data/Vaccine/COVID-19/data_tables/vaccine_data/us_data/time_series/vaccine_data_us_timeline.csv'
raw_data_global = './data/Vaccine/COVID-19/data_tables/vaccine_data/global_data/vaccine_data_global.csv'

ts_vaccine_global = './data/Vaccine/COVID-19/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_global.csv'
ts_vaccine_doses_global = './data/Vaccine/COVID-19/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_doses_admin_global.csv'

print("Start importing single data vaccine in US", raw_data_us)
total_confirmed_vaccine_us = Start.import_data(raw_data_us)

print("Start importing single data vaccine in the World", raw_data_global)
total_confirmed_vaccine_global = Start.import_data(raw_data_global)

print("Start importing time series data vaccine in the World", ts_vaccine_global)
time_series_vaccine_global = Start.import_data(ts_vaccine_global)

# print("Start importing time series data vaccine doses admin in the World", ts_vaccine_doses_global)
# time_series_admin_vaccine_global = Start.import_data(ts_vaccine_doses_global)
