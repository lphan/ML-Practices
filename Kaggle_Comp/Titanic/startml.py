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


import configparser
import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer


class StartML(object):
    """
    Description: StartML - Start Machine Learning
    Import parameters from config.ini and execute basic pre_processing operations
    Return all the basic statistics about data
    Start: 
    jupyter notebook    
    -> from startml import *
    -> info_help    
    """

    def __init__(self):
        pass

    # key words arguments which contains all values from config.ini
    kwargs = {}

    @staticmethod
    def _get_arguments():
        """
        read config-parameters from file config.ini
        """
        config = configparser.ConfigParser()
        config.read('config.ini')

        data_path_1 = config['paths']['data_path_1']
        data_path_2 = config['paths']['data_path_2']

        nan_drop_col = config.getboolean('StartML', 'replace_nan_drop_column')
        nan_drop_row = config.getboolean('StartML', 'replace_nan_drop_row')
        nan_zero = config.getboolean('StartML', 'replace_nan_zero')
        nan_mean = config.getboolean('StartML', 'replace_nan_mean')
        nan_mean_neighbors = config.getboolean('StartML', 'replace_nan_mean_neighbors')

        StartML.kwargs.update({"data_path_1": data_path_1,
                               "data_path_2": data_path_2,
                               "nan_drop_col": nan_drop_col,
                               "nan_drop_row": nan_drop_row,
                               "nan_zero": nan_zero,
                               "nan_mean": nan_mean,
                               "nan_mean_neighbors": nan_mean_neighbors})

    @classmethod
    def groupby_columns(cls, data, columns):
        """
        operation group_by and count the frequency of all single variables in every columns
        :param data:
        :param columns:
        :return:
        """
        # tbd
        pass

    @classmethod
    def get_value_column_index(cls, data, column_name, row_id):
        """
        given data, column_name and row_id
        return value at row_id of column
        :param data:
        :param column_name:
        :param row_id:
        :return: value at row_id of the given column
        """
        # return data.column_name[row_id]  # (short-way)
        return data[column_name][data[column_name].index[row_id]]
        
    @classmethod
    def nan_columns(cls, data):
        """
        return name of all columns which have NaN_value
        :param data:
        :return: key_true
        """
        kc = data.isnull().any()
        key_true = [key for key, value in kc.iteritems() if value]

        return key_true

    @classmethod
    def nan_rows(cls, data):
        """
        return all rows containing NaN values in type DataFrame
        :param data:
        :return: nan_rows
        """
        return data[data.isnull().any(axis=1)]

    @classmethod
    def feature_engineering(cls, data):
        # tbd
        pass

    @classmethod
    def mean_neighbors(cls, data, column, row_id):
        """
        compute mean value of value at row_id with values from its above and lower neighbors.
        if the above neighbor is NaN, it jumps to higher position
        :param column:
        :param row_id:
        :return:
        """
        above_rid = row_id - 1
        while np.isnan(StartML.get_value_column_index(data, column, above_rid)):
            above_rid = above_rid - 1
        above_val = StartML.get_value_column_index(data, column, above_rid)

        lower_rid = row_id + 1
        while np.isnan(StartML.get_value_column_index(data, column, lower_rid)):
            lower_rid = lower_rid + 1
        lower_val = StartML.get_value_column_index(data, column, lower_rid)

        return np.mean([lower_val, above_val])

    @classmethod
    def pre_processing_columns(cls, data):
        """
        pre_processing columns based on information given in the config.ini
        :param data:
        :return: data after pre-processing
        """
        nan_cols = cls.nan_columns(data)

        # Drop the columns where all elements are nan
        data = data.dropna(axis=1, how='all')

        if nan_cols and StartML.kwargs['nan_drop_col']:

            # drop all nan_columns, axis : {0 or 'index (rows)', 1 or 'columns'}
            return data.drop(nan_cols, axis=1)

        elif nan_cols and StartML.kwargs['nan_zero']:

            # convert nan_value in column into zero_value (WARNING: columns in dtypes float64, int64), axis=1
            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)
            return data

        elif nan_cols and StartML.kwargs['nan_mean']:
            # convert nan into mean_value of column (WARNING: only suitable for columns in dtypes float64, int64)
            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    data[nan_col] = data[nan_cols].groupby(nan_col).mean()
            return data

        else:
            print("Data in columns safe!")
            return data

    @classmethod
    def pre_processing_rows(cls, data):
        """
        pre_processing rows based on information given in the config.ini
        :param data:
        :return: data after pre-processing
        """
        nan_cols = cls.nan_columns(data)
        nan_rows = cls.nan_rows(data)

        data = data.drop_duplicates()

        # if not nan_rows.empty:
        #     # Return DataFrame with duplicate rows removed
        #     data = data.drop_duplicates()

        if not nan_rows.empty and StartML.kwargs['nan_drop_row']:
            # Drop the rows where all elements are nan
            data = data.dropna(axis=0, how='all')

            # Drop row if it does not have at least two values that are **not** NaN
            return data.dropna(thresh=2)

        elif not nan_rows.empty and StartML.kwargs['nan_zero']:

            # convert nan in row into zero_value, axis=0
            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)
            return data

        elif not nan_rows.empty and StartML.kwargs['nan_mean']:

            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    # data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)

                    # compute the mean of neighbor-values, option: 'most_frequent', 'median'
                    imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
                    imputer = imputer.fit(data[nan_col].values.reshape(1, -1))
                    data[nan_col] = imputer.transform(data[nan_col].values.reshape(1, -1))[0]
            return data

        elif not nan_rows.empty and StartML.kwargs['nan_mean_neighbors']:

            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    # for row_id in range(len(data[nan_col])):
                    #     if np.isnan(StartML.get_value_column_index(data, nan_col, row_id)):
                    #         data[nan_col][row_id] = StartML.mean_neighbors(data, nan_col, row_id)
                    data[nan_col] = [StartML.mean_neighbors(data, nan_col, row_id)
                                     if np.isnan(StartML.get_value_column_index(data, nan_col, row_id))
                                     else data[nan_col][row_id] for row_id in range(len(data[nan_col]))
                                     ]

            return data

        else:
            print("Data in rows safe, please check StartML.nan_columns again!")
            return data

    @staticmethod
    def summary(data):
        """
        Show all basic information
        """
        print("\nData Columns: ", data.columns, "\n")
        print("Missing values in Data: \n")
        print(data.isnull().sum(), "\n")
        print("data.head(10): \n")
        print(data.head(10), "\n")
        print("data.info(): \n")
        print(data.info(), "\n")
        print("data.describe(): \n")
        print(data.describe(), "\n")

    @staticmethod
    def run():
        """
        Read data from data_set .csv and convert them into Pandas Data Frame
        """
        StartML._get_arguments()
        data_path_1 = pd.read_csv(StartML.kwargs['data_path_1'])
        data_path_2 = pd.read_csv(StartML.kwargs['data_path_2'])

        return data_path_1, data_path_2


train_data, test_data = StartML.run()

info_help = {
            "info_help": StartML.__name__,
            "StartML.kwargs": "Show key words arguments from config.ini",
            "StartML.summary(data)": StartML.summary.__doc__,
            "StartML.pre_processing_columns(data)": StartML.pre_processing_columns.__doc__,
            "StartML.pre_processing_rows(data)": StartML.pre_processing_rows.__doc__,
            "StartML.nan_columns(data)": StartML.nan_columns.__doc__,
            "StartML.nan_rows(data)": StartML.nan_rows.__doc__,
            "train_data": train_data.__class__,
            "test_data": test_data.__class__
            }

StartML.__doc__
