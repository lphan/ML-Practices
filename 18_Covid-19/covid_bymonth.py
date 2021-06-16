import calendar
from covid import *
from vaccine_import import *

files_by_month = [[str(month)+'-'+str(calendar.monthrange(year, month)[1])+'-'+str(year)+'.csv' if month>9 else str(0)+str(month)+'-'+str(calendar.monthrange(year, month)[1])+'-'+str(year)+'.csv' for month in range(1, 13)] for year in [2020, 2021]]

# flatten all lists into one list
files_by_month = [y for x in files_by_month for y in x]

# add current month by finding the data last day. parse the month, current_data_this_month = the difference last day with previous month
files_by_month.append(updated_date+'.csv')

# query data in year 2020 (check if file exists, then import data)
data_by_month = [Start.import_data(path_folder+pathfile) for pathfile in files_by_month if os.path.isfile(path_folder+pathfile)]

countries_confirmed_month = pd.DataFrame(columns=countries)
countries_fatalities_month = pd.DataFrame(columns=countries)
countries_recovered_month = pd.DataFrame(columns=countries)

'''
Number of cases (confirmed, deaths, recovered) by month
'''
for month in range(len(data_by_month)):   
    if 'Country/Region' in data_by_month[month].columns:
        countries_confirmed_month= countries_confirmed_month.append(data_by_month[month].groupby('Country/Region').sum().transpose().loc['Confirmed'], ignore_index=True)
        countries_fatalities_month= countries_fatalities_month.append(data_by_month[month].groupby('Country/Region').sum().transpose().loc['Deaths'], ignore_index=True)
        countries_recovered_month= countries_recovered_month.append(data_by_month[month].groupby('Country/Region').sum().transpose().loc['Recovered'], ignore_index=True)
    else:
        countries_confirmed_month= countries_confirmed_month.append(data_by_month[month].groupby('Country_Region').sum().transpose().loc['Confirmed'], ignore_index=True)
        countries_fatalities_month= countries_fatalities_month.append(data_by_month[month].groupby('Country_Region').sum().transpose().loc['Deaths'], ignore_index=True)
        countries_recovered_month= countries_recovered_month.append(data_by_month[month].groupby('Country_Region').sum().transpose().loc['Recovered'], ignore_index=True)
        
countries_confirmed_month.fillna(0, inplace=True)
countries_fatalities_month.fillna(0, inplace=True)
countries_recovered_month.fillna(0, inplace=True)
