# setup absolute path to location of package Starts and config-file
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

# from numba import jit
# from numba import njit
import seaborn as sns
from Starts.start import *
from Starts.startml import *
from Starts.startvis import *
from matplotlib.pylab import rcParams

# Import general data (without US-recovered)
Start._arguments()

# Import global data
path_confirmed_global = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
path_deaths_global = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
path_recovered_global = './COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

# Import World data
path_folder = Start.kwargs['folder_path']

# Import USA data (used to retrieve number of recovered in USA)
path_us_folder = './COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/'

if path_folder:
    print("Start importing World-data ", path_folder)
    files = sorted(os.listdir(path_folder)) 
    
    # # clean up unused files
    files.pop(0), files.pop()

    formats = [file.split('.')[1] for file in files]
    files_new = [file.split('.')[0] for file in files]
    files_new = sorted(files_new, key=lambda date: datetime.strptime(date, "%m-%d-%Y"))
    files = [files_new[i]+'.'+formats[i] for i in range(len(files_new))]

    data = [Start.import_data(path_folder+pathfile) for pathfile in files]

if path_us_folder:
    print("Start importing USA-data ", path_us_folder)
    files_us = sorted(os.listdir(path_us_folder))

    # clean up unused files
    files_us.pop() # data.pop(0), data.pop(), data_us.pop()

    formats_us = [file_us.split('.')[1] for file_us in files_us]
    files_new_us = [file_us.split('.')[0] for file_us in files_us]
    files_new_us = sorted(files_new_us, key=lambda date: datetime.strptime(date, "%m-%d-%Y"))
    files_us = [files_new_us[i]+'.'+formats_us[i] for i in range(len(files_new_us))]

    data_us = [Start.import_data(path_us_folder+file_us) for file_us in files_us]

if Start.kwargs['data_path']:
    path_filename = Start.kwargs['data_path']
    print("Start importing single data ", path_filename)
    sdata = Start.import_data(path_filename)
else: 
    print("No Data_Path or Folder_Path is given")

# Persist data in memory to allow future computations faster (only apply for dask-object)
if Start.kwargs['pandas_type'] is False:
    data = [dat.persist() for dat in data]

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
data_us = [data_us[day].fillna(0) for day in x_dat_us]

# TODO: fix population 'Dominica Republic' (>10m) 180k covid vs 'Dominica' (76k) 107 covid, there are two location with Dominica