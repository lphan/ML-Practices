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
# %matplotlib inline

# CHINA: Processing NaN value
y_dat_cn = [StartML.searchByValue(data[i], column='Country/Region', value='Mainland China')['Confirmed'].values 
         for i in range(len(data))]

y_dat_cn[0] = np.array([y for y in y_dat_cn[0] if not np.isnan(y)])
y_dat_cn[1] = np.array([y for y in y_dat_cn[1] if not np.isnan(y)])
y_dat_cn[2] = np.array([y for y in y_dat_cn[2] if not np.isnan(y)])
y_dat_cn = [sum(y) for y in y_dat_cn]

# GERMANY: Processing empty value
y_dat_de = [StartML.searchByValue(data[i], column='Country/Region', value='Germany')['Confirmed'].values
             for i in range(len(data))]
y_dat_de = [0 if y.size == 0 else y[0] for y in y_dat_de]

# ITALY: Processing empty value
y_dat_it = [StartML.searchByValue(data[i], column='Country/Region', value='Italy')['Confirmed'].values
             for i in range(len(data))]
y_dat_it = [0 if y.size == 0 else y[0] for y in y_dat_it]

# KOREA: Processing
y_dat_kn = [StartML.searchByValue(data[i], column='Country/Region', value='South Korea')['Confirmed'].values
             for i in range(len(data))]
y_dat_kn = [y[0] for y in y_dat_kn]

# JAPAN: Processing
y_dat_jap = [StartML.searchByValue(data[i], column='Country/Region', value="Japan")['Confirmed'].values
             for i in range(len(data))]
y_dat_jap = [y[0] for y in y_dat_jap]

x_dat = np.arange(len(data))
