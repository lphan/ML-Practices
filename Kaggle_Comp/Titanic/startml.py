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
    Description: StartML
    Import parameters from config.ini and execute basic pre_processing
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
    def get_arguments():
        """
        read config-parameters from file config.ini
        """
        config = configparser.ConfigParser()
        config.read('config.ini')

        data_path_1 = config['paths']['data_path_1']
        data_path_2 = config['paths']['data_path_2']

        nan_drop_col = config['StartOps']['replace_nan_column_drop']
        nan_drop_row = config['StartOps']['replace_nan_row_drop']
        nan_zero = config['StartOps']['replace_nan_zero']
        nan_mean = config['StartOps']['replace_nan_mean']

        StartML.kwargs.update({"data_path_1": data_path_1,
                               "data_path_2": data_path_2,
                               "nan_drop_col": nan_drop_col,
                               "nan_drop_row": nan_drop_row,
                               "nan_zero": nan_zero,
                               "nan_mean": nan_mean})

    # TODO: methods
    # operation group_by and count the frequency of all single variables in every columns
    # -> startvis: implement basic plots (scatter, histograms, bar charts)
    # -> startmod: implement basic correlations between variables

    @classmethod
    def get_index(cls, data, column_name, row_id):
        """
        given data, column_name and row_id
        return value at row_id of column
        :param data:
        :param column_name:
        :param row_id:
        :return:
        """
        return data[column_name][data[column_name].index[row_id]]

    # @classmethod
    # def mean_neighbors(cls, data, idx):
    #     """
    #     return mean value from neighbor-elements (+1, -1)
    #     :param data:
    #     :return:
    #     """
    #     pass

    # @classmethod
    # def object_columns(cls, data):
    #     """
    #     return all columns with type object
    #     :param data:
    #     :return:
    #     """
    #     return

    @classmethod
    def nan_columns(cls, data):
        """
        return name of all columns which have NaN_value
        :param data:
        :return:
        """
        kc = data.isnull().any()
        key_true = [key for key, value in kc.iteritems() if value]

        return key_true

    @classmethod
    def nan_rows(cls, data):
        """
        return all rows containing NaN values in type DataFrame
        :param data:
        :return:
        """
        return data[data.isnull().any(axis=1)]

    @classmethod
    def feature_engineering(cls, data):
        pass

    @classmethod
    def operations_base(cls, data):

        nan_cols = cls.nan_columns(data)
        if nan_cols:
            data.dropna(axis=1, how='all')  # Drop the columns where all elements are nan

            if StartML.kwargs['nan_drop_col']:
                # drop all nan_columns, axis : {0 or 'index (rows)', 1 or 'columns'}
                return data.drop(nan_cols, axis=1)

            elif StartML.kwargs['nan_zero']:
                # convert nan in column into zero_value (WARNING: only suitable for columns in dtypes float64, int64)
                for nan_col in nan_cols:
                    if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                        data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0, axis=1)
                return data

            elif StartML.kwargs['nan_mean']:
                # convert nan into mean_value of column (WARNING: only suitable for columns in dtypes float64, int64)
                for nan_col in nan_cols:
                    if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                        data[nan_col] = data[nan_cols].groupby(nan_col).mean()
                return data
        else:
            print("Data in columns safe!")

        nan_rows = cls.nan_rows(data)
        if nan_rows:
            data.dropna(axis=0, how='all')  # Drop the rows where all elements are nan

            if StartML.kwargs['nan_drop_row']:
                # Keep only the rows with max 2 non-na values
                return data.dropna(thresh=2)

            elif StartML.kwargs['nan_zero']:
                # convert nan in row into zero_value, axis=0
                for nan_col in nan_cols:
                    if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                        data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)
                return data

            elif StartML.kwargs['nan_mean']:
                for nan_col in nan_cols:
                    if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                        # data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)

                        # compute the mean of neighbor-values, option: 'most_frequent', 'median'
                        imputer = Imputer(missing_values='NaN', strategy='mean', axis=1)
                        imputer = imputer.fit(data[nan_col].values.reshape(1, -1))
                        data[nan_col] = imputer.transform(data[nan_col].values.reshape(1, -1))[0]
                return data
        else:
            print("Data in rows safe, please check again with StartML.nan_columns ..!")

        return data

    @staticmethod
    def summary_base(data):
        """
        Show all basic information
        """
        print("\n", data.columns, "\n")
        print(data.head(10), "\n")
        print(data.info(), "\n")
        print(data.describe(), "\n")

    @staticmethod
    def run():
        """
        Read data from data_set .csv and convert them into Pandas Data Frame
        """
        StartML.get_arguments()
        data_path_1 = pd.read_csv(StartML.kwargs['data_path_1'])
        data_path_2 = pd.read_csv(StartML.kwargs['data_path_2'])

        return data_path_1, data_path_2


train_data, test_data = StartML.run()

# info_help is dict_object which contain information about all important objects and methods used in this data_analytics
info_help = {
            "info_help": StartML.__name__,
            "StartML.kwargs": "Show key words arguments from config.ini",
            "StartML.summary_base(..data..)": "Show basic information about data set\n",
            "StartML.operation_base(..data..)": "Execute basic operations on data set\n",
            "StartML.nan_columns(..data..)": "True if data has NaN value and show NaN_columns",
            "StartML.nan_rows(..data..)": "Show all NaN rows",
            "train_data": train_data.__class__,
            "test_data": test_data.__class__
            }
