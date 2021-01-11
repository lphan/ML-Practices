# # ANSWERING the AD-HOC QUESTIONS
# # REWRITE COVID in pandas-DataFrame format
# # setup absolute path to location of package Starts and config-file 
# from inspect import getsourcefile
# import os.path as path, sys

# current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
# sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

# from Starts.start import *
# from Starts.startml import *
# from Starts.startvis import *
# from covid_import import *
# from matplotlib.pylab import rcParams

# rcParams['figure.figsize'] = 20, 6

# ''' Data Preprocessing '''
# # collect all data into list all countries of tuple (country, confirmed), (country, fatalities), (country, recovered)
# countries = sdata['Country_Region'].unique()

# ''' Number of all infected countries changed by day '''

# # filter column by name and convert Pandas frame to Numpy Array
# infected_countries_earliest = np.unique(data[0][data[0]['Confirmed']>0].filter(regex=("Country.*")).values)
# infected_countries_latest = np.unique(data[-1][data[-1]['Confirmed']>0].filter(regex=("Country.*")).values)

# num_infected_countries = [len(np.unique(data[i][data[i]['Confirmed']>0].filter(regex=("Country.*")).values)) for i in range(len(data))]

# # get basic data
# dataconfirmed = [dict() for i in range(len(data))]
# datafatal = [dict() for i in range(len(data))]
# datarecovered = [dict() for i in range(len(data))]

# for i in range(len(data)):
#     # print("Day: ", i)
#     col = data[i].filter(like='Country').columns[0]
    
#     for country in infected_countries_latest:
#         dataconfirmed[i][country] = y_dat_confirmed[country][i]
#         datafatal[i][country] = y_dat_deaths[country][i]
#         datarecovered[i][country] = y_dat_recovered[country][i]

# # create 3x data frames (columns = all countries, rows = all days) for confirmed, deaths, and recovered
# pdConfirm = pd.DataFrame(data=dataconfirmed, columns=infected_countries_latest)
# pdDeaths = pd.DataFrame(data=datafatal, columns=infected_countries_latest)
# pdRecovered = pd.DataFrame(data=datarecovered, columns=infected_countries_latest)

# # Select top and bottom values
# totalConfirmed = pdConfirm.tail(1).values.sum()
# totalFatal = pdDeaths.tail(1).values.sum()
# totalRecovered = pdRecovered.tail(1).values.sum()

# lastday=len(data)-1

# # Top 10 highest
# top10confirmed = pdConfirm.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
# top10confirmed['RatioByTotal_in_%']=[np.round(top10confirmed.loc[country].values[0]/totalConfirmed*100, 4) for country in top10confirmed.index]

# top10fatal = pdDeaths.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
# top10fatal['RatioByTotal_in_%']=[np.round(top10fatal.loc[country].values[0]/totalFatal*100, 4) for country in top10fatal.index]

# top10recovered = pdRecovered.tail(1).transpose().sort_values(by=[lastday], ascending=False).head(10)
# top10recovered['RatioByTotal_in_%']=[np.round(top10recovered.loc[country].values[0]/totalRecovered*100, 4) for country in top10recovered.index]

# # Top 10 lowest
# top10confirmed_lowest = pdConfirm.tail(1).transpose().sort_values(by=[lastday], ascending=True)
# top10confirmed_lowest['RatioConfirmedByPopulation_in_%']=[np.round(top10confirmed_lowest.loc[country].values[0]/sdata[sdata['Country_Region']==country]['Population'].values[0] *100, 4) for country in top10confirmed_lowest.index]
# top10confirmed_lowest['population']= [sdata[sdata['Country_Region']==country]['Population'].values[0] for country in top10confirmed_lowest.index]

# top10fatal_lowest = pdDeaths.tail(1).transpose().sort_values(by=[lastday], ascending=True)
# top10fatal_lowest['RatioFatalByConfirmed_in_%']=[np.round(top10fatal_lowest.loc[country].values[0]/ pdConfirm[country].tail(1).values[0] *100, 4) for country in top10fatal_lowest.index]
# top10fatal_lowest['Confirmed']= [pdConfirm[country].tail(1).values[0] for country in top10fatal_lowest.index]

# top10recovered_lowest = pdRecovered.tail(1).transpose().sort_values(by=[lastday], ascending=True)
# top10recovered_lowest['RatioRecoveredByConfirmed_in_%']=[np.round(top10recovered_lowest.loc[country].values[0]/ pdConfirm[country].tail(1).values[0] *100, 4) for country in top10recovered_lowest.index]
# top10recovered_lowest['Confirmed']= [pdConfirm[country].tail(1).values[0] for country in top10recovered_lowest.index]