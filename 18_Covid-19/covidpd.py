# ANSWERING the AD-HOC QUESTIONS
# REWRITE COVID in pandas-DataFrame format
# setup absolute path to location of package Starts and config-file 
from inspect import getsourcefile
import os.path as path, sys

current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
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

# x-axis for plot
x_dat = np.arange(len(data))

# collect all data into list all countries of tuple (country, confirmed), (country, fatalities), (country, recovered)
countries = sdata['Country_Region'].unique()

''' Number of all infected countries changed by day '''

# filter column by name and convert Pandas frame to Numpy Array
infected_countries_earliest = np.unique(data[0][data[0]['Confirmed']>0].filter(regex=("Country.*")).values)
infected_countries_latest = np.unique(data[-1][data[-1]['Confirmed']>0].filter(regex=("Country.*")).values)

num_infected_countries = [len(np.unique(data[i][data[i]['Confirmed']>0].filter(regex=("Country.*")).values)) for i in range(len(data))]

pdConfirm = pd.DataFrame(columns=infected_countries_latest)
pdDeaths = pd.DataFrame(columns=infected_countries_latest)
pdRecovered = pd.DataFrame(columns=infected_countries_latest)

dataconfirmed = dict()
datafatal = dict()
datarecovered = dict()

for i in range(len(data)):
    # print("Day: ", i)
    col = data[i].filter(like='Country').columns[0]
    
    dataconfirmed[i]=[data[i].loc[data[i][col]==country]['Confirmed'].sum() for country in infected_countries_latest]    
    datafatal[i]=[data[i].loc[data[i][col]==country]['Deaths'].sum() for country in infected_countries_latest]
    datarecovered[i]=[data[i].loc[data[i][col]==country]['Recovered'].sum() for country in infected_countries_latest]

pdConfirm=pdConfirm.from_dict(dataconfirmed, orient='index', columns=infected_countries_latest)
pdDeaths=pdDeaths.from_dict(datafatal, orient='index', columns=infected_countries_latest)
pdRecovered=pdRecovered.from_dict(datarecovered, orient='index', columns=infected_countries_latest)

totalConfirmed = pdConfirm.tail(1).values.sum()
totalFatal = pdDeaths.tail(1).values.sum()
totalRecovered = pdRecovered.tail(1).values.sum()

lastday=len(data)-1

# Top 10 highest
top10confirmed = pdConfirm.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
top10confirmed['RatioByTotal_in_%']=[np.round(top10confirmed.loc[country].values[0]/totalConfirmed*100, 4) for country in top10confirmed.index]

top10fatal = pdDeaths.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
top10fatal['RatioByTotal_in_%']=[np.round(top10fatal.loc[country].values[0]/totalFatal*100, 4) for country in top10fatal.index]

top10recovered = pdRecovered.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
top10recovered['RatioByTotal_in_%']=[np.round(top10recovered.loc[country].values[0]/totalRecovered*100, 4) for country in top10recovered.index]

# Top 10 lowest
top10confirmed_lowest = pdConfirm.tail(1).transpose().sort_values(by=[lastday], ascending=True)
top10confirmed_lowest['RatioConfirmedByPopulation_in_%']=[np.round(top10confirmed_lowest.loc[country].values[0]/sdata[sdata['Country_Region']==country]['Population'].values[0] *100, 4) for country in top10confirmed_lowest.index]
top10confirmed_lowest['population']= [sdata[sdata['Country_Region']==country]['Population'].values[0] for country in top10confirmed_lowest.index]

top10fatal_lowest = pdDeaths.tail(1).transpose().sort_values(by=[lastday], ascending=True)
top10fatal_lowest['RatioFatalByConfirmed_in_%']=[np.round(top10fatal_lowest.loc[country].values[0]/ pdConfirm[country].tail(1).values[0] *100, 4) for country in top10fatal_lowest.index]
top10fatal_lowest['Confirmed']= [pdConfirm[country].tail(1).values[0] for country in top10fatal_lowest.index]

top10recovered_lowest = pdRecovered.tail(1).transpose().sort_values(by=[lastday], ascending=True)
top10recovered_lowest['RatioRecoveredByConfirmed_in_%']=[np.round(top10recovered_lowest.loc[country].values[0]/ pdConfirm[country].tail(1).values[0] *100, 4) for country in top10recovered_lowest.index]
top10recovered_lowest['Confirmed']= [pdConfirm[country].tail(1).values[0] for country in top10recovered_lowest.index]