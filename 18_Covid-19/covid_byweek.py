from covid import *

''' 
Total of infected cases, fatalities, recovered in the world changed by week 
'''
# @jit(nopython=True)
# @njit
def numberByWeeks(data):
    weeks = list()

    # First week
    weeks.append((1, data[4]))

    # start from second week
    week = 2
    for i in range(11, len(x_dat), 7):
        weeks.append((week, data[i]))
        week = week + 1
    
    # Add the last day of current week to sums
    weeks.append((week, data[-1]))

    return weeks

confirmedByWeek = numberByWeeks(data=totalconfirmed_by_day)
deathsByWeek = numberByWeeks(data=totalfatalities_by_day)
recoveredByWeek = numberByWeeks(data=totalrecovered_by_day)