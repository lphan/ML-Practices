#!/usr/bin/env python3
#
# Copyright (c) 2018
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
    def _arguments():
        """
        read config-parameters from file config.ini
        """
        config = configparser.ConfigParser()
        config.read('config.ini')

        data_path_1 = config['paths']['data_path_1']
        data_path_2 = config['paths']['data_path_2']

        exclude_obj_col = config.getboolean('StartML', 'exclude_object_column')
        nan_drop_col = config.getboolean('StartML', 'nan_drop_column')
        nan_drop_row = config.getboolean('StartML', 'nan_drop_row')
        nan_zero = config.getboolean('StartML', 'nan_zero')
        nan_mean = config.getboolean('StartML', 'nan_mean')
        nan_mean_neighbors = config.getboolean('StartML', 'nan_mean_neighbors')

        StartML.kwargs.update({"data_path_1": data_path_1,
                               "data_path_2": data_path_2,
                               "drop_obj_col": exclude_obj_col,
                               "nan_drop_col": nan_drop_col,
                               "nan_drop_row": nan_drop_row,
                               "nan_zero": nan_zero,
                               "nan_mean": nan_mean,
                               "nan_mean_neighbors": nan_mean_neighbors})

    @classmethod
    def find_idx_max_value(cls, data):
        for i, v in enumerate(data):
            if v == max(data):
                break
        return max(data), i

    @classmethod
    def find_value(cls, data, rows_id, column_name=''):
        """
        given data, column_name and row_id
        return value at row_id of column
        :param data: Pandas-DataFrame
        :param column_name:
        :param rows_id: list type as list of rows_id
        :return: list of tuple (column, row, value)
        """
        # return data.column_name[row_id]  # (short-way)
        # return data.iloc[row_id, column_id], data.loc[row_id, column_label]
        # return [(column_name, data[column_name][data[column_name].index[row_id])]
        try:
            if not column_name:
                return [(row_id, data.iloc[row_id]) for row_id in rows_id]
            else:
                return [(row_id, column_name, data.at[row_id, column_name]) for row_id in rows_id]
        except KeyError:
            print('Rows_id is out of range')
            return []

    @classmethod
    def idx_reset(cls, data):
        return data.reset_index(drop=True, inplace=True)

    @classmethod
    def idx_columns(cls, data):
        """
        return a list of tuple (column, index, label_type)
        :param data:
        :return: list of tuple (column, column_idx, type's column)
        """
        return [(col, data.columns.get_loc(col), data.dtypes[col]) for col in data.columns]

    @classmethod
    def group_by_columns(cls, data, columns, label_groupby, func=None):
        """
        execute operation group_by on columns by label_groupby
        :param data: Pandas-DataFrame
        :param columns: list of columns need to be grouped
        :param label_groupby: need to be one of the given columns
        :param func:
        :return: DataFrameGroupBy object (which can be used to compute further)
        """
        grouped = data[columns].groupby(label_groupby)
        if func is None:
            return grouped
        else:
            return grouped.aggregate(func)

    @classmethod
    def lookup_value(cls, data, value):
        """
        find all values in data frame
        :param data: Pandas-DataFrame
        :param value (can be either int, float or object)
        :return: list of tuple (row_id, 'column_name')
        """
        # tbd: value in regex*

        # identify all columns with the same type as value
        if isinstance(type(value), str):
            # type(value) == str:
            print('Object')
            search_columns = [col for col in data.columns if data.dtypes[data.columns.get_loc(col)] == 'O']
        else:
            search_columns = [col for col in data.columns if data.dtypes[data.columns.get_loc(col)] == int or
                       data.dtypes[data.columns.get_loc(col)] == float]

        # loop on these columns to look up value
        result = []

        for idx, rows in data.iterrows():
            result = result + [(idx, col) for col in search_columns if rows[col] == value]

        return result

    @classmethod
    def mean_neighbors(cls, data, row_id, column):
        """
        compute mean value of value at row_id with values from its above and lower neighbors.
        if the above neighbor is NaN, it jumps to higher position
        similarly if the lower neighbor is NaN, it jumps to higher position.
        :param row_id:
        :param column:
        :return: mean value of neighbors
        """
        # if row_id is min (e.g. 0), the closest 'non_NaN' value will be taken
        if row_id == min(data.index):
            # print("MIN INDEX")
            higher_rid = row_id + 1
            while np.isnan(data.at[higher_rid, column]):
                # print(lower_rid, data.at[lower_rid, column])
                higher_rid = higher_rid + 1
            # print(data.at[lower_rid, column])
            return data.at[higher_rid, column]

        # if row_id is max (e.g. length of data), the closest 'non_NaN' value will be taken
        elif row_id == max(data.index):
            # print("MAX INDEX")
            lower_rid = row_id - 1

            while np.isnan(data.at[lower_rid, column]):
                lower_rid = lower_rid - 1

            return data.at[lower_rid, column]
        else:
            # print("MIN MAX INDEX")
            lower_rid = row_id - 1

            while np.isnan(data.at[lower_rid, column]) and lower_rid > min(data.index):
                lower_rid = lower_rid - 1
            if np.isnan(data.at[lower_rid, column]):
                lower_val = 0
            else:
                lower_val = data.at[lower_rid, column]

            higher_rid = row_id + 1
            while np.isnan(data.at[higher_rid, column]) and higher_rid < max(data.index):
                higher_rid = higher_rid + 1
            if np.isnan(data.at[higher_rid, column]):
                higher_val = 0
            else:
                higher_val = data.at[higher_rid, column]

            # print(lower_val, higher_val)
            return np.mean([lower_val, higher_val])

    @classmethod
    def nan_columns(cls, data):
        """
        return name of all columns which have NaN_value
        :param data: Pandas-DataFrame
        :return: list of all possible NaN_column(s)
        """
        nan_bool = data.isnull().any()
        if isinstance(data, pd.DataFrame):
            key_true = [key for key, value in nan_bool.iteritems() if value]
            return key_true
        else:
            return nan_bool

    @classmethod
    def nan_rows(cls, data, nan=True):
        """
        return all rows containing NaN values in type DataFrame
        :param data: Pandas-DataFrame
        :param nan: Boolean-input True to search for NaN values, False for not_NaN
        :return: data with all possible found NaN_rows or not_NaN_rows (if nan=False)
        """
        if isinstance(data, pd.DataFrame) and nan:
            return data[data.isnull().any(axis=1)]

        elif isinstance(data, pd.DataFrame) and not nan:
            return data[data.notnull().any(axis=1)].dropna(axis=0, how='any')

        else:
            return data.isnull().any()

    @classmethod
    def process_nan_columns(cls, data):
        """
        pre_processing columns based on information given in the config.ini
        :param data: Pandas-DataFrame
        :return: data after pre-processing
        """

        nan_cols = cls.nan_columns(data)

        # Drop the columns where all elements are nan
        data = data.dropna(axis=1, how='all')

        if nan_cols and StartML.kwargs['drop_obj_col']:
            # drop all columns with type object
            return data.select_dtypes(exclude=object)

        elif nan_cols and StartML.kwargs['nan_drop_col']:

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
                    # data[nan_col] = data[nan_cols].groupby(nan_col).mean()
                    data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=np.mean(data[nan_col]))
            return data

        else:

            return data

    @classmethod
    def process_nan_rows(cls, data):
        """
        pre_processing rows based on information given in the config.ini
        :param data: Pandas-DataFrame
        :return: data after pre-processing
        """
        # tbd: improve performance

        nan_cols = cls.nan_columns(data)
        nan_rows = cls.nan_rows(data)

        # drop the duplicates rows
        data = data.drop_duplicates()

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
                    #     if np.isnan(data.at[row_id, nan_col]):
                    #         data[nan_col][row_id] = StartML.mean_neighbors(data, row_id, nan_col)
                    data[nan_col] = [StartML.mean_neighbors(data, row_id, nan_col)
                                     if np.isnan(data.at[row_id, nan_col]) else data[nan_col][row_id]
                                     for row_id in range(len(data[nan_col]))
                                     ]
            return data
        else:
            return data

    @classmethod
    def process_nan_simply(cls, data):
        """
        simply process all nan-value by replacing with 'Unknown'
        :param data: Pandas-DataFrame
        :return: data after preprocessing
        """
        for col in StartML.nan_columns(data):
            data[col] = data[col].fillna('Unknown')
        return data

    @staticmethod
    def obj_num_convert(data):
        """
        convert data from object-type into numeric type
        :param data: Pandas-DataFrame
        :return:
        """
        for col in data.columns:
            try:
                # raise exception
                data[col] = pd.to_numeric(data[col], errors='raise')
            except ValueError:
                # set as NaN
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data

    @staticmethod
    def nan_summary(data):
        """
        display summary about all NaN values in data
        :param data: Pandas-DataFrame
        :return:
        """
        print("Nans_columns: \n{}".format(StartML.nan_columns(data)))
        print("Nans_rows: \n{}".format(len(StartML.nan_rows(data))))

    @staticmethod
    def summary(data):
        """
        Show all basic information about data set
        """
        print("\nData Columns: {}".format(data.columns), "\n")
        print("Missing values in Data: \n{}".format(data.isnull().sum()), "\n")
        print("data.head(10): \n{}".format(data.head(10)), "\n")
        print("data.info(): \n{}".format(data.info()), "\n")
        print("data.describe(): \n{}".format(data.describe()), "\n")
        print(StartML.nan_summary(data))

    @staticmethod
    def run():
        """
        Read data from data_set .csv and convert them into Pandas Data Frame
        """
        StartML._arguments()
        if StartML.kwargs['data_path_1']:
            data_path_1 = pd.read_csv(StartML.kwargs['data_path_1'])
        else:
            data_path_1 = ''
        if StartML.kwargs['data_path_2']:
            data_path_2 = pd.read_csv(StartML.kwargs['data_path_2'])
        else:
            data_path_2 = ''

        return data_path_1, data_path_2

    @staticmethod
    def info_help():

        return {
            "info_help_StartML": StartML.__name__,
            "StartML.kwargs": "Show key words arguments from config.ini",
            "StartML.summary(data)": StartML.summary.__doc__,
            "StartML.pre_processing_columns(data)": StartML.process_nan_columns.__doc__,
            "StartML.pre_processing_rows(data)": StartML.process_nan_rows.__doc__,
            "StartML.nan_columns(data)": StartML.nan_columns.__doc__,
            "StartML.nan_rows(data)": StartML.nan_rows.__doc__,
            "train_data": train_data.__class__,
            "test_data": test_data.__class__
        }


train_data, test_data = StartML.run()

# Remove leading and trailing spaces in columns
if isinstance(train_data, pd.DataFrame):
    # if not train_data.empty:
    train_data.columns = [col.strip() for col in train_data.columns]

if isinstance(test_data, pd.DataFrame):
    # if not test_data.empty:
    test_data.columns = [col.strip() for col in test_data.columns]

info_ml = StartML.info_help()

StartML.__doc__
