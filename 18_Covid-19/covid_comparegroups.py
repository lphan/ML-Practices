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
