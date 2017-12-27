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
Description: others download media-data
'''
import configparser
import pandas as pd
import params
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.pylab import rcParams
import timeit


def run():
    argument = get_arguments()
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %H:%M:%S')

    path = argument['dataset_path']

    hw = pd.read_csv(path, index_col=0, parse_dates={'datetime': ['Datum', 'Zeit']}, date_parser=dateparse)
    
    # convert to time series
    ts = hw['W [cm]']

    return hw, ts


def get_arguments():
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    now_water_level = config['input']['now_water_level']
    now_time = config['input']['now_time']
    predict_time = config['input']['predict_time']
    water_level_chain = config['input']['water_level_chain']

    pre_process = config.getboolean('process', 'pre_processing')
    classify = config.getboolean('process', 'classify')
    predict = config.getboolean('process', 'predict')
    rate_of_change = config.getboolean('process', 'rate_of_change')
    filter_data = config.getboolean('process', 'filter_data')

    image_path = config['paths']['image_path']
    dataset_path = config['paths']['dataset_path']

    kwargs = {"now_water_level": now_water_level, "now_time": now_time, "predict_time": predict_time,
              "water_level_chain": water_level_chain,
              "pre_process": pre_process, "classify": classify, "predict": predict,
              "rate_of_change": rate_of_change, "filter_data": filter_data, "image_path": image_path, "dataset_path": dataset_path }

    return (kwargs)


def getMaxMin(ts):
    max1993 = ts['1993'][ts['1993'] == max(ts['1993'])]
    min1993 = ts['1993'][ts['1993'] == min(ts['1993'])]

    max1994 = ts['1994'][ts['1994'] == max(ts['1994'])]
    min1994 = ts['1994'][ts['1994'] == min(ts['1994'])]

    max2013 = ts['2013'][ts['2013'] == max(ts['2013'])]
    min2013 = ts['2013'][ts['2013'] == min(ts['2013'])]

    max2016 = ts['2016'][ts['2016'] == max(ts['2016'])]
    min2016 = ts['2016'][ts['2016'] == min(ts['2016'])]
    return max1993, min1993, max1994, min1994, max2013, min2013, max2016, min2016

hw, ts = run()
max1993, min1993, max1994, min1994, max2013, min2013, max2016, min2016 = getMaxMin(ts)

# print (test)
# print (i)
