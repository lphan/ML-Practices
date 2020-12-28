# setup absolute path to location of package Starts and config-file
import seaborn as sns
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from numba import jit
from numba import njit
from Starts.start import *
from Starts.startml import *
from Starts.startvis import *
from matplotlib.pylab import rcParams

# Import general data (without US-recovered)
Start._arguments()

path_folder = Start.kwargs['folder_path']

if Start.kwargs['folder_path']:
    print("Start importing files in folder")   
    data, files = Start.import_folder(path_folder)

if Start.kwargs['data_path']:
    path_filename = Start.kwargs['data_path']
    print("Start importing single data ", path_filename)
    sdata = Start.import_data(path_filename)
else: 
    print("No Data_Path or Folder_Path is given")

# Persist data in memory to allow future computations faster (only apply for dask-object)
if Start.kwargs['pandas_type'] is False:
    data = [dat.persist() for dat in data]

# Import USA data (used to retrieve number of recovered in USA)
path_us_folder = './COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/'
print("Start importing USA-data ", path_us_folder)
data_us, files_us = Start.import_folder(path_us_folder)

# Import global data 
path_confirmed_global = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
path_deaths_global = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
path_recovered_global = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

print("Start importing single data ", path_confirmed_global)
total_confirmed = Start.import_data(path_confirmed_global)

print("Start importing single data ", path_deaths_global)
total_deaths = Start.import_data(path_deaths_global)

print("Start importing single data ", path_recovered_global)
total_recovered_without_US = Start.import_data(path_recovered_global)

# x-axis (days) for plot
x_dat = np.arange(len(data))
x_dat_us = np.arange(len(data_us))

# Pre-Processing: fill all NaN with 0
data = [data[day].fillna(0) for day in x_dat]