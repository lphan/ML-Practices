from covid import *

'''
Total comparison increasing by day in 
Western_culture (10 countries: US Germany Italy Spain France UK Swiss Netherland Austria Belgium) 
and
Estern_culture (10 countries:  China Korea Japan Malaysia Indonesia Thailand Philippine Singapore Taiwan Vietnam)
'''

first_group = ['Italy', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria', 'Belgium', 'Luxembourg']
# first_group = ['Norway', 'Sweden', 'Denmark', 'Finland']
second_group = ['China', 'Korea, South', 'Japan', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Singapore', 'Taiwan*', 'Vietnam']

# first_group: female, second_group: male
# first_group = ['Germany', 'Taiwan*', 'New Zealand', 'Iceland', 'Finland', 'Norway', 'Denmark']
# second_group = ['US', 'Brazil', 'United Kingdom', 'Russia', 'Italy', 'Spain', 'France']

first_group_population = [country_pop_dict[country] for country in first_group]
second_group_population = [country_pop_dict[country] for country in second_group]

comparision = pd.DataFrame(index=x_dat, columns=['FirstGroup_confirmed', 'SecondGroup_confirmed', 'FirstGroup_fatalities', 'SecondGroup_fatalities', 'FirstGroup_recovered', 'SecondGroup_recovered'])
comparision.fillna(0, inplace=True)

for idx in x_dat:
    comparision['FirstGroup_confirmed'].loc[idx] = sum(countries_confirmed[first_group].loc[idx])
    comparision['SecondGroup_confirmed'].loc[idx] = sum(countries_confirmed[second_group].loc[idx])
    
    comparision['FirstGroup_fatalities'].loc[idx] = sum(countries_fatalities[first_group].loc[idx])
    comparision['SecondGroup_fatalities'].loc[idx] = sum(countries_fatalities[second_group].loc[idx])
    
    comparision['FirstGroup_recovered'].loc[idx] = sum(countries_recovered[first_group].loc[idx])
    comparision['SecondGroup_recovered'].loc[idx] = sum(countries_recovered[second_group].loc[idx])


# # total cases by days in 10 EU countries and 10 ASIA countries
# firstgroup_byDay = []
# secondgroup_byDay = []

# # total fatalities by days in 10 EU countries and 10 ASIA countries
# firstgroup_deaths_byDay = []
# secondgroup_deaths_byDay = []

# # total recovered by days in 10 EU countries and 10 ASIA countries
# firstgroup_rec_byDay = []
# secondgroup_rec_byDay = []

# # TODO: convert to Data frame
# # iterate all days
# for i in x_dat:
#     firstgroup_byDay.append(
#         [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Confirmed'].values
#          for ec in first_group])
#     secondgroup_byDay.append(
#         [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Confirmed'].values
#          for ac in second_group])

#     firstgroup_deaths_byDay.append(
#         [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Deaths'].values
#          for ec in first_group])
#     secondgroup_deaths_byDay.append(
#         [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Deaths'].values
#          for ac in second_group])

#     firstgroup_rec_byDay.append(
#         [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ec)['Recovered'].values
#          for ec in first_group])
#     secondgroup_rec_byDay.append(
#         [StartML.searchByValue(data[i], try_keys=['Country_Region', 'Country/Region'], value=ac)['Recovered'].values
#          for ac in second_group])

# # fill empty data by 0
# # @jit(nopython=True)
# # @njit
# def getfill(fillbyday):
#     return [[np.array([0]) if fi.size == 0 else fi for fi in fill] for fill in fillbyday]
    
# fill_firstgroup_byDay = getfill(fillbyday=firstgroup_byDay)
# fill_secondgroup_byDay = getfill(fillbyday=secondgroup_byDay)

# fill_firstgroup_byDay_temp = []
# fill_secondgroup_byDay_temp = []

# fill_firstgroup_fatal_byday_temp = []
# fill_secondgroup_fatal_byday_temp = []

# fill_firstgroup_recovered_byday_temp= []
# fill_secondgroup_recovered_byday_temp = []

# # iterate all days
# for i in x_dat:
#     fill_firstgroup_byDay_temp.append([sum(fill_firstgroup) for fill_firstgroup in fill_firstgroup_byDay[i]])
#     fill_secondgroup_byDay_temp.append([sum(fill_secondgroup) for fill_secondgroup in fill_secondgroup_byDay[i]])

#     fill_firstgroup_fatal_byday_temp.append([sum(fill_firstgroup) for fill_firstgroup in firstgroup_deaths_byDay[i]])
#     fill_secondgroup_fatal_byday_temp.append([sum(fill_secondgroup) for fill_secondgroup in secondgroup_deaths_byDay[i]])

#     fill_firstgroup_recovered_byday_temp.append([sum(fill_firstgroup) for fill_firstgroup in firstgroup_rec_byDay[i]])
#     fill_secondgroup_recovered_byday_temp.append([sum(fill_secondgroup) for fill_secondgroup in secondgroup_rec_byDay[i]])


# # Computation the total cases in EU and ASIA (infected cases, fatalities, recovered) 
# firstgroup_total = []
# secondgroup_total = []

# firstgroup_deaths_total = []
# secondgroup_deaths_total = []

# firstgroup_rec_total = []
# secondgroup_rec_total = []

# for i in x_dat:
#     firstgroup_total.append(sum(fill_firstgroup_byDay_temp[i]))
#     secondgroup_total.append(sum(fill_secondgroup_byDay_temp[i]))

#     firstgroup_deaths_total.append(sum(fill_firstgroup_fatal_byday_temp[i]))
#     secondgroup_deaths_total.append(sum(fill_secondgroup_fatal_byday_temp[i]))

#     firstgroup_rec_total.append(sum(fill_firstgroup_recovered_byday_temp[i]))
#     secondgroup_rec_total.append(sum(fill_secondgroup_recovered_byday_temp[i]))