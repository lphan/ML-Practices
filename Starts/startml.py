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
    (statistics, groupby, reduce, etc.) on datasets

    Start: 
        jupyter notebook
        -> from startml import *
        -> info_help
    """
    # init keywords arguments
    kwargs = {}

    def __init__(self):
        pass

    @staticmethod
    def _arguments():
        """
        read config-parameters from local file config.ini
        """
        # key words arguments which contains all values from config.ini
        try:
            config = configparser.ConfigParser()
            config.read('config.ini')
        except IOError:
            print("Error open file config.ini")
            return

        # pass data from config file to local var
        data_path = config['paths']['data_path']

        exclude_obj_col = config.getboolean('StartML', 'exclude_object_column')
        nan_drop_col = config.getboolean('StartML', 'nan_drop_column')
        nan_drop_row = config.getboolean('StartML', 'nan_drop_row')
        nan_zero = config.getboolean('StartML', 'nan_zero')
        nan_mean = config.getboolean('StartML', 'nan_mean')
        nan_mean_neighbors = config.getboolean('StartML', 'nan_mean_neighbors')

        StartML.kwargs.update({"data_path": data_path,
                               "drop_obj_col": exclude_obj_col,
                               "nan_drop_col": nan_drop_col,
                               "nan_drop_row": nan_drop_row,
                               "nan_zero": nan_zero,
                               "nan_mean": nan_mean,
                               "nan_mean_neighbors": nan_mean_neighbors})
        print("local_kwargs", StartML.kwargs)

    @classmethod
    def convert_time_series(cls, data, time_column, format=None, add_day=False):
        """
        convert dataset into time_series dataset

        :param data: pandas.core.frame.DataFrame
        :param time_column:
        :param format: default True if time_column is in date_time format, False if in millisecond
        :return: new_data
        """
        if not format:
            data.index = pd.to_datetime(data.pop(time_column))
        else:
            data.index = pd.to_datetime(data.pop(time_column), unit='ms', format=format)

        if add_day:
            data['day'] = [t.weekday() for t in data.index]
        data = data.sort_index()
        return data

    @classmethod
    def infobyTime(cls, data, time_range, func):
        """
        compute values in period windows
        (e.g. by every certain 'Monday' days| 'Jan' months| years, every number N [days| weeks| month| years]),
        apply for TimeSeries data

        Useful operations:
        1. find mean value by day (7 days per week) in a period time
        2. find mean value by week in a period time
        3. find mean value by month in a period time

        :param data:
        :param time_range:
        :param func: functions from Numpy e.g. np.mean, np.std, etc.
        :return:
        """
        pass

    @classmethod
    def head_dict(cls, data, h=5):
        """
        get the first elements in dict
        :param data:
        :return:
        """
        return list(data.items())[0:h]

    @classmethod
    def intersect_dict(cls, dict1, dict2):
        """
        return the common key_value pairs
        :param dict1:
        :param dict2:
        :return:
        """
        return [item for item in list(dict1.items()) if item[1] in dict2.values()]

    @classmethod
    def findMaxMinValueDict(cls, data, ma=True):
        """
        return the data (key, value) with max/ min value
        (room for improvement of performance)
        :param data: dict-type
        :param ma: default max if True (min if False)
        :return: data: dict-type
        """
        if ma:
            return dict([(k, v) for k, v in data.items() if v == max(data.values())])
        else:
            return dict([(k, v) for k, v in data.items() if v == min(data.values())])

    @classmethod
    def getkeyby_value(cls, data, value):
        """
        return a list of pair key_value which contain the certain value

        :param data: dict_type
        :param value:
        :return:
        """
        # return [l for l in list(dict) if l[1] == value]
        return [k for k in data if data.get(k) == value]

    @classmethod
    def find_value(cls, data, rows_id, column_name=''):
        """
        given data, column_name and row_id
        return value at row_id of column

        :param data: pandas.core.frame.DataFrame
        :param column_name: find on specific column
        :param rows_id: list of rows_id
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
    def find_idx_max_value(cls, data):
        for i, v in enumerate(data):
            if v == max(data):
                break
        return max(data), i

    @classmethod
    def find_idx(cls, data, column, values):
        """
        find all rows_idx which have the same values located at column of data
        :param data: pandas.core.frame.DataFrame
        :param values: the given values
        :param column: column_feature where to look for values
        :return: DataFrame object
        """

        # # init empty DataFrame
        # x = pd.DataFrame(columns=data.columns)
        #
        # # append/ fill DataFrame with values
        # for v in values:
        #     x = x.append(pd.DataFrame(data[data[column] == v], columns=data.columns))
        return [pd.DataFrame(data[data[column] == v], columns=data.columns) for v in values]

    @classmethod
    def idx_reset(cls, data):
        return data.reset_index(drop=True, inplace=True)

    @classmethod
    def idx_columns(cls, data):
        """
        return a list of tuple (column, index, label_type)

        :param data: pandas.core.frame.DataFrame
        :return: list of tuple (column, column_idx, type's column)
        """
        return [(col, data.columns.get_loc(col), data.dtypes[col]) for col in data.columns]

    @classmethod
    def parallelby_func(cls, data, func):
        """
        execute operation paralellization (MultiThreading, MultiProcessing) as wrapper on func
        using Spark
        :param data:
        :return:
        """
        pass

    @classmethod
    def groupby_columns(cls, data, columns, groupby_label, func=None):
        """
        execute operation group_by on columns by label_groupby
        e.g. compute mean value by column 'day'
            StartML.groupby_columns(data, columns=['values'], groupby_label=['day'], func=np.mean)

        References:
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html

        :param data: pandas.core.frame.DataFrame
        :param columns: list of columns need to be grouped
        :param groupby_label: need to be one of the given columns
        :param func: e.g. np.mean, np.median, np.mode, etc.
        :return: dict-object (which can be used to compute further)
        """
        grouped = data.groupby(groupby_label)
        if func is None:
            return grouped.groups
        else:
            # return grouped.aggregate(func)
            return grouped[columns].agg(func)

    @classmethod
    def groupby_rows(cls, data, groupby_label, func=None):
        """
        groupby values which have the same rows_id identified by column

        :param data: pandas.core.frame.DataFrame
        :param groupby_label:
        :param func:
        :return:
        """

        grouped = data[groupby_label].unique()
        new_list = {}
        for item in grouped:
            key = data[data[groupby_label]==item]
            value = key.drop(groupby_label, axis=1).to_dict()
            new = {item: value}
            new_list.update(new)

        return new_list

    @classmethod
    def reduceby_rows(cls, data, operations):
        """
        reduce number of rows (map_reduce, map needs 'list(mapped_obj)', reduce needs 'import functools')
        or (DataFrame_apply)

        :param data: pandas.core.frame.DataFrame
        :param rows:
        :param operation: add, subtract, multiplication, mean etc.
        :return:
        """
        pass

    @classmethod
    def compute_correlation_dataframe(cls, df1, df2=None):
        """
        Compute pairwise-Correlation based on Pearson Correlation between 2 data frames or data frame itself

        References:
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corrwith.html
            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

        :param df1: pandas.core.frame.DataFrame
        :param df2: pandas.core.frame.DataFrame
        :return: pandas.core.frame.DataFrame or pandas.core.frame.DataFrame
        """
        if not df2:
            return df1.corr()
        else:
            return df1.corrwith(df2)

    @classmethod
    def generate_correlation_matrix(cls, data):
        """
        Dimensionality reduction using truncated SVD

        References:
            https://en.wikipedia.org/wiki/Singular-value_decomposition
            http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.corrcoef.html

        :param data:
        :return:
        """
        pass

    @classmethod
    def detect_outliers(cls, data):
        """
        Algorithm: setup a threshold for error (maximal error).
        if the computed error exceeds the threshold, then the data point will be listed as outlier (anomaly detection).
        Choose to remove (clean) or neutralize using Minkowski-method

        Algorithm functions in combination with plot-visual and observation.
        Outliers are the points outside the range [(Q1-1.5 IQR), (Q3+1.5 IQR)]

        References:
            https://www.neuraldesigner.com/blog/3_methods_to_deal_with_outliers
            http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm

        :param data: pandas.core.series.Series
        :return:
        """
        total_outliers = {}

        # Step1: find all outliers in every columns
        for col in data.columns:
            # min value
            min_value = data[col].describe()['min']

            # lower_quartile = np.percentile(nonan_data['Fare'].values, 25)
            q1_lower_quartile = data[col].describe()['25%']

            # median = np.median(nonan_data['Fare'].values)
            q2_median = data[col].describe()['50%']

            # upper_quartile = np.percentile(nonan_data['Fare'].values, 75)
            q3_upper_quartile = data[col].describe()['75%']

            # max value
            max_value = data[col].describe()['max']

            IQR = abs(q3_upper_quartile - q1_lower_quartile)
            IQR = 1.5*IQR
            lower_bound = q1_lower_quartile - IQR
            upper_bound = q3_upper_quartile + IQR

            # print(min_value, q1_lower_quartile, q2_median, q3_upper_quartile, max_value)
            # print(lower_bound, upper_bound)
            outliers = [v for v in data[col].values if v < lower_bound or v > upper_bound]
            # print(col, outliers)
            # print(len(outliers))
            total_outliers[col] = outliers
            # print()

        # tbd Step2: find all row_idx of these outliers points which appear in all columns
        #
        # then find the occurrences > 2
        #
        # tbd Step3: remove or keep it by modifying its values
        return total_outliers

    @classmethod
    def anomaly_detection(cls, data):
        """
        Detect anomaly features
        Application like: Fraud detection, monitoring machines in data center

        Transform non_Gaussian features into Gaussian features using e.g. function log(..)
        Using:
            1. Original model
                manually create additional features to capture anomalies (unusual combinations of values)
            2. Multivariate Gaussian Distribution,
                calculate population mean, variance matrix Sigma
                automatically capture correlations between features

        References:
            https://en.wikipedia.org/wiki/Anomaly_detection
            https://www.youtube.com/watch?v=g2YBWQnqOpw&index=90&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN
            https://www.youtube.com/watch?v=JjB58InuTqM&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=94
            https://www.youtube.com/watch?v=YRS-IB3vCow&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=95

        :param data:
        :return:
        """
        pass

    @classmethod
    def anomaly_feature_detect(cls, data):
        """

        :param data:
        :return:
        """
        pass

    @classmethod
    def applyby_func(cls, data, columns, ops):
        """
        Apply func on certain columns (to change or update values)
        :param data: pandas.core.frame.DataFrame
        :param columns: list of columns which will be updated by operation ops
        :param ops: mean, median, mode
        :return:
        """
        if ops in ['mean', 'median', 'mode']:
            for col in columns:
                if ops is 'mean':
                    # try using df.loc[row_index,col_indexer] = value with lambda
                    data[col] = data[col].apply(lambda x: data[col].mean())
                elif ops is 'median':
                    data[col] = data[col].apply(lambda x: data[col].median())
                else:
                    data[col] = data[col].apply(lambda x: data[col].mode())
        else:
            print("Ops is not valid, only accept one of operation mean, median, mode")

        return data

    @classmethod
    def filterby_rows(cls, data, func):
        """
        filter out all the values (rows) which are considered "No need" for dataset
        to reduce the unnecessary data processing (using DataFrame_apply)

        :param data:
        :param func:
        :return:
        """
        return data.drop(data[func].index)

    @classmethod
    def unionby_rows(cls, data, func):
        """
        union all rows together with the pre-defined values (using DataFrame_apply)
        :param data:
        :param func:
        :return:
        """
        # data.apply(func, )
        pass

    @classmethod
    def mergeby_data(cls, data1, data2):
        """
        union data1 and data2 and filter out all duplicates (using DataFrame_merge)
        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def joinby_data(cls, data1, data2):
        """
        proceed different join_operations (left, right, inner, outer) on dataset (using DataFrame_merge)
        (similar as mergeby_data)
        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def intersectionby_data(cls, data1, data2):
        """
        proceed operation intersect to get the common part between data1 and data2 (using DataFrame_merge)
        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def mergeby_data(cls, data, rows, columns):
        """
        merge values from rows and columns together
        :param data:
        :param rows:
        :param columns:
        :return:
        """
        pass

    @classmethod
    def countby_values(cls, data, value=None):
        """
        count value by key (using DataFrame_to_dict to convert DataFrame into dict_type)
        :param data: pandas.core.series.Series
        :param value: value in column which need to be counted
        :return: list of tuple
        """
        # use np.count_nonzero or np.sum
        if not value:
            return dict([(c, np.count_nonzero((data == c).values)) for c in data.unique()])
        else:
            return dict([(value, np.count_nonzero((data == value).values))])

    @classmethod
    def orderby_kv(cls, data, keyvalue):
        """
        sort (ascending, descending) of rows, columns (using DataFrame.sort_values)
        :param data:
        :param keyvalue:
        :return:
        """
        pass

    @classmethod
    def lookup_value(cls, data, value, tup=True):
        """
        find all values in data frame

        :param data: pandas.core.frame.DataFrame
        :param value (can be either int, float or object)
        :param tup (True will return as tuple with column, False will return a list of row_id)
        :return: list of tuple (row_id, 'column_name')
        """
        # tbd: value in regex*

        # identify all columns with the same type as value
        if isinstance(type(value), str):
            # type(value) == str:
            # print('Object')
            search_columns = [col for col in data.columns if data.dtypes[data.columns.get_loc(col)] == 'O']
        else:
            search_columns = [col for col in data.columns if data.dtypes[data.columns.get_loc(col)] == 'int32' or
                              data.dtypes[data.columns.get_loc(col)] == 'float64']

        # loop on these columns to look up value
        result = []
        # print(search_columns)
        if not tup:
            for idx, rows in data.iterrows():
                result = result + [idx for col in search_columns if rows[col] == value]
            # return np.array(result)
        else:
            for idx, rows in data.iterrows():
                result = result + [(idx, col) for col in search_columns if rows[col] == value]
                # print(result)

        return np.array(result)

    @classmethod
    def mean_neighbors(cls, data, row_id, column):
        """
        compute mean value of value at row_id with values from its above and lower neighbors.
        if the above neighbor is NaN, it jumps to higher position
        similarly if the lower neighbor is NaN, it jumps to higher position.

        :param data: pandas.core.frame.DataFrame
        :param row_id: index row
        :param column: column name
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

        :param data: pandas.core.frame.DataFrame
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
        :param data: pandas.core.frame.DataFrame
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
    def pop_rows(cls, data, idx, inplace=True):
        """
        get all rows with idx out of data
        :param data: pandas.core.frame.DataFrame
        :param idx:
        :return:
        """
        try:
            pop_rows = [data.loc[id] for id in idx if id in data.index]
            if inplace:
                data.drop(idx, axis=0, inplace=True)
            else:
                data.drop(idx, axis=0, inplace=False)

        except ValueError:
            print("ValueError, index ", idx, "does not exist")
            import sys
            sys.exit(1)

        return pd.DataFrame(data=pop_rows, columns=data.columns)

    @classmethod
    def process_nan_columns(cls, data):
        """
        pre_processing columns based on information given in the config.ini
        :param data: pandas.core.frame.DataFrame
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

        :param data: pandas.core.frame.DataFrame
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
    def process_nan_simply(cls, data, nan_column=None):
        """
        simply process all nan-value by replacing with 'Unknown'

        :param data: pandas.core.frame.DataFrame
        :param nan_column: single NaN_column
        :return: data after preprocessing
        """
        if nan_column:
            # process single NaN_column
            data[nan_column] = data[nan_column].fillna('Unknown')
        else:
            # process multiple NaN_columns
            for col in StartML.nan_columns(data):
                data[col] = data[col].fillna('Unknown')
        return data

    @classmethod
    def merge_df(cls, data, feature):
        """
        merge
        :param data: list of data frames
        :param feature: common feature in between data frames
        :return:
        """
        if not data:
            return

        if len(data) == 1:
            return data[0]

        dat = data.pop()
        for _ in range(len(data)):
            item = data.pop()
            tmp = dat.merge(item, how='left', on=[feature])
            dat = tmp

        return dat

    @staticmethod
    def obj_num_convert(data):
        """
        convert data from object-type into numeric type
        :param data: pandas.core.frame.DataFrame
        :return: the converted data in numeric_type
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
    def resampling_interpolate(data, option):
        """
        Resampling (Upsampling and Downsampling) involves changing the frequency of your time series observations
        Object must have a datetime-like index
        Using the interpolation to fill the gap inside data

        Reference:
            http://benalexkeen.com/resampling-time-series-data-with-pandas/
            https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
            http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.resample.html
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#resampling

        :param data: pandas.core.frame.DataFrame
        :param option: apply to time series index via Day/ Month/ Year
        :return:
        """
        pass

    @staticmethod
    def nan_summary(data):
        """
        display summary about all NaN values in data

        :param data: pandas.core.frame.DataFrame
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
        run data processing pipeline
        """
        pass

    @staticmethod
    def import_data():
        """
        Read data from data_set .csv and convert them into Pandas Data Frame and append them into a list

        References:
            https://docs.python.org/3/library/codecs.html#standard-encodings
        """
        StartML._arguments()
        if not StartML.kwargs:
            return

        df = []
        try:
            if StartML.kwargs['data_path']:
                paths = StartML.kwargs['data_path'].split(',')
                for path in paths:

                    # remove space before and after string
                    path = path.strip()

                    if path.endswith('.xlsx') or path.endswith('.xls'):
                        data_exl = pd.read_excel(path)
                        df.append(data_exl)
                    elif path.endswith('.json'):
                        data_json = pd.read_json(path, low_memory=False)
                        df.append(data_json)
                    elif path.endswith('.csv'):
                        data_csv = pd.read_csv(path, low_memory=False)
                        df.append(data_csv)
                    else:
                        print('Unknown format')
                        return
            else:
                print("Data is not given")
                return

        except FileNotFoundError as fe:
            print("\nFileNotFoundError, data does not exist", fe)
            # raise
            import sys
            sys.exit(1)

        return df

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
            "data": data.__class__
        }


data = []
for dat in StartML.import_data():
    if isinstance(dat, pd.DataFrame):
        dat.columns = [col.strip() for col in dat.columns]
        data.append(dat)

info_ml = StartML.info_help()

StartML.__doc__
