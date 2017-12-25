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
# import json
# import io


def execute():
    
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
