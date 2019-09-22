#!/usr/bin/env python3
#
# Copyright (c) 2014-2015
#
# This software is licensed to you under the GNU General Public License,
# version 2 (GPLv2). There is NO WARRANTY for this software, express or
# implied, including the implied warranties of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. You should have received a copy of GPLv2
# along with this software; if not, see
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.

__author__ = 'Long Phan'

'''
Description: Import parameters from config.ini and execute data pre_processing 
'''
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

from datetime import datetime
from matplotlib.pylab import rcParams
from sklearn.preprocessing import Imputer


def get_arguments():
    """
    read config-parameters from file config.ini
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    data_set2014_path = config['paths']['data_set2014_path']
    data_set2015_path = config['paths']['data_set2015_path']
    data_dictionary_path = config['paths']['data_dictionary_path']
    
    kwargs = {"data_set2014_path": data_set2014_path, "data_set2015_path": data_set2015_path,
              "data_dictionary_path": data_dictionary_path}

    return kwargs


def group_columns_mean(data, columns, column):
    """
    return mean value of the given columns grouped by the given column
    :param data:
    :param columns:
    :param column:
    :return:
    """
    return data[columns].groupby(column).mean()


def get_data_state_mean_est(data):
    return group_columns_mean(data, columns=['STATE_CODE', 'LOW_ESTIMATE', 'HIGH_ESTIMATE'], column='STATE_CODE')


def get_data_state_mean_est_spec(data, state_code, estimate='LOW_ESTIMATE'):
    state_mean = get_data_state_mean_est(data)

    if estimate == 'HIGH_ESTIMATE' and state_code in state_mean.index:
        return state_mean.at[state_code, 'HIGH_ESTIMATE']
    elif state_code in state_mean.index:
        return get_data_state_mean_est(data).at[state_code, 'LOW_ESTIMATE']
    else:
        print("No valid data about state ", state_code)
        return


def get_state_list(data):
    return get_data_state_mean_est(data).index


def nan_columns(dat):
    """
    return name of all columns which have NaN_value
    :param dat:
    :return:
    """
    kc = dat.isnull().any()
    # print(kc.keys())
    key_true = [key for key, value in kc.iteritems() if value]

    return key_true


def nan_rows(data):
    """
    return all rows containing NaN values in type DataFrame
    :param data:
    :return:
    """
    return data[data.isnull().any(axis=1)]


def process_missing_data(data):

    # Replace missing value as NaN by 0 following requirement 'LOW_ESTIMATE'
    data[['LOW_ESTIMATE']] = data[['LOW_ESTIMATE']].replace(np.NaN, 0)

    # Find all nans_row similarly as nans = lambda df: df[df.isnull().any(axis=1)]
    # and replace missing value as NaN by mean of the neighboring states following requirement 'high_estimate'
    # TODO: apply(lambda ...)
    nans_dat = nan_rows(data)
    for index, row in nans_dat.iterrows():
        new_value = get_data_state_mean_est_spec(data, nans_dat['STATE_CODE'].values[0], 'HIGH_ESTIMATE')
        data.loc[index, 'HIGH_ESTIMATE'] = new_value

    return data


def run():
    """
    Read data from data_set and convert to time series format
    """
    argument = get_arguments()

    data2014 = pd.read_csv(argument['data_set2014_path'])
    data2015 = pd.read_csv(argument['data_set2015_path'])
    dictionary = pd.read_csv(argument['data_dictionary_path'])

    return process_missing_data(data2014), process_missing_data(data2015), dictionary


def get_max_min(data):
    """
    return max value and min value of every year in type Series
    """
    max_year_high_est = (data[['YEAR', 'HIGH_ESTIMATE']]).max()
    max_year_low_est = (data[['YEAR', 'LOW_ESTIMATE']]).max()

    min_year_high_est = (data[['YEAR', 'HIGH_ESTIMATE']]).min()
    min_year_low_est = (data[['YEAR', 'LOW_ESTIMATE']]).min()

    return max_year_high_est, max_year_low_est, min_year_high_est, min_year_low_est


d2014, d2015, states = run()

d2014_states = d2014.merge(states, how='left', on=['STATE_CODE', 'COUNTY_CODE'])
d2015_states = d2015.merge(states, how='left', on=['STATE_CODE', 'COUNTY_CODE'])
