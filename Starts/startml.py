#!/usr/bin/env python3
#
# Copyright (c) 2019
#
# This software is licensed to you under the GNU General Public License,
# version 2 (GPLv2). There is NO WARRANTY for this software, express or
# implied, including the implied warranties of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. You should have received a copy of GPLv2
# along with this software; if not, see
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.

__author__ = 'Long Phan'


import configparser
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import fnmatch
from Starts import *
from Starts.start import *
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from numpy import cov
from numpy import percentile

class StartML(Start):
    """
    Description: StartML - Start Machine Learning
    Import parameters from config.ini and execute basic pre_processing operations
    (statistics, groupby, reduce, etc.) on datasets

    Start: 
        jupyter notebook
        -> from startml import *
        -> info_help
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def convert_time_series(cls, data, time_column, format=None, add_day=False):
        """
        Description: convert dataset into time_series dataset

        :param data: pandas.core.frame.DataFrame
        :param time_column: contains time-series data
        :param format: default True if time_column is in date_time format, False if in millisecond
        :return: new_data
        """
        if not format:
            # TODO: wrong convert time day
            data.index = pd.to_datetime(data.pop(time_column), unit='ms')
        else:
            data.index = pd.to_datetime(data.pop(time_column), unit='ms', format=format)

        if add_day:
            data['day'] = [t.weekday() for t in data.index]
        data = data.sort_index()
        return data

    @classmethod
    def infobyTime(cls, data, time_range, func):
        """
        Description: compute values in period windows
        (e.g. by every certain 'Monday' days| 'Jan' months| years, every number N [days| weeks| month| years]),
        apply for TimeSeries data

        Useful operations:
            1. find mean value by day (7 days per week) in a period time
            2. find mean value by week in a period time
            3. find mean value by month in a period time

        :param data: Time series data (Pandas format)
        :param time_range: from .. to
        :param func: functions from Numpy e.g. np.mean, np.std, etc.
        :return:
        """
        pass

    @classmethod
    def head_dict(cls, data, headElems=5):
        """
        get the first elements in dict

        :param data: pandas.core.frame.DataFrame
        :param headElems: the first elements to get out (default: 5)
        :return:
        """
        return list(data.items())[0:headElems]

    @classmethod
    def intersect_dict(cls, dict1, dict2):
        """
        Description: return the common key_value pairs

        :param dict1: data in dict-format
        :param dict2: data in dict-format
        :return:
        """
        return [item for item in list(dict1.items()) if item[1] in dict2.values()]

    @classmethod
    def findMaxMinValueDict(cls, data, maxVal=True):
        """
        Description: return the data (key, value) with max/ min value (tbd for improvement of performance)

        :param data: dict-type
        :param maxVal: default max if True (min if False)
        :return: data: dict-type
        """
        if maxVal:
            return dict([(k, v) for k, v in data.items() if v == max(data.values())])
        else:
            return dict([(k, v) for k, v in data.items() if v == min(data.values())])

    @classmethod
    def getKeyByValue(cls, data, value):
        """
        return a list of pair key_value which contain the certain value

        :param data: dict_type
        :param value: 
        :return:
        """
        # return [l for l in list(dict) if l[1] == value]
        return [k for k in data if data.get(k) == value]

    @classmethod
    def findValue(cls, data, rows_id, column_name=''):
        """
        Description: given data, column_name and row_id and return value at row_id of column

        :param data: pandas.core.frame.DataFrame
        :param column_name: find on specific column (default: empty)
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
        """
        Description: find index of maximal value in the given data

        :param data:
        :return:
        """
        for i, v in enumerate(data):
            if v == max(data):
                break
        return max(data), i

    @classmethod
    def find_idx(cls, data, column, values):
        """
        Description: find all rows_idx which have the same values located at column of data

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
        """
        Description: reset the index of data

        :param data:
        :return:
        """
        return data.reset_index(drop=True, inplace=True)

    @classmethod
    def idx_columns(cls, data):
        """
        Description: return a list of tuple (column, index, label_type)

        :param data: pandas.core.frame.DataFrame
        :return: list of tuple (column, column_idx, type's column)
        """
        return [(col, data.columns.get_loc(col), data.dtypes[col]) for col in data.columns]

    @classmethod
    def parallelby_func(cls, data, func):
        """
        Description: execute operation paralellization (MultiThreading, MultiProcessing) as wrapper on func using Spark

        :param data:
        :return:
        """
        pass

    @classmethod
    def groupby_columns(cls, data, groupby_label, columns=None, func=None, classification=False):
        """
        Description: execute operation group_by on columns by label_groupby
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
        # if classification:
        #     print(data.groupby(groupby_label).size())

        if func is None:
            # return grouped.groups
            return grouped
        else:
            # return grouped.aggregate(func)
            return grouped[columns].agg(func)

    @classmethod
    def groupby_rows(cls, data, groupby_label, func=None):
        """
        Description: group by values which have the same rows_id identified by column

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
        Description: reduce number of rows (map_reduce, map needs 'list(mapped_obj)', reduce needs 'import functools')
        or (DataFrame_apply)

        :param data: pandas.core.frame.DataFrame
        :param rows:
        :param operation: add, subtract, multiplication, mean etc.
        :return:
        """
        pass

    
    @classmethod
    def comp_corr_nonGaussian(cls, data1, data2, spearmanr=True):
        """
        Description:
            quantifying the association between variables with a non-Gaussian distribution
            by calculating the Spearmanr's (default) rank correlation coefficient or Kendall's rank 
            between two variables (on an ordinal scale of measurement)
        :param data1: Numpy array
        :param data2: Numpy array
        :return coef (range between -1, 1), p (range between 0,1: >0.05 or <0.05)
        """
        if spearmanr:
            # calculate spearman's correlation and p-value
            coef, p = spearmanr(data1, data2)
        else:
            # calculate Kendall's correlation and p-value
            coef, p = kendalltau(data1, data2)
        print('Correlation coefficient: %.3f' % coef)
        # interpret the significance
        alpha = 0.05
        if p > alpha:
            print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
        else:
            print('Samples are correlated (reject H0) p=%.3f' % p)
        return coef, p

    @classmethod
    def comp_corr_df(cls, df1, df2=None):
        """
        Description: 
            Compute pairwise-Correlation based on Pearson Correlation between 2 data frames or data frame itself, 
            describe the relationship between two variables and whether they might change together.

        References:
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corrwith.html
            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

        :param df1: pandas.core.frame.DataFrame
        :param df2: pandas.core.frame.DataFrame
        :return: pandas.core.frame.DataFrame or pandas.core.frame.DataFrame
        """
        pd.set_option('display.width', 100)
        pd.set_option('precision', 3)
        if not df2:
            return df1.corr(method='pearson')
        else:
            return df1.corrwith(df2)

    @classmethod
    def comp_corr_col(cls, x1, x2):
        """
        Description: 
            Compute correlation between 2 columns x1 and x2
            First check whether x_column and y_column have the same type numpy ndarray and same size
        :param x: numpy array
        :param y: numpy array
        """
        if (type(x1) == type(x2) and len(x1) == len(x2) and type(x2) is np.ndarray and type(x1) is np.ndarray):            
            corr, p = pearsonr(x1.reshape(-1), x2)
            print("Pearson correlation: %.3f" % corr)
            # interpret the significance
            alpha = 0.05
            if p > alpha:
                print('No correlation (fail to reject H0)')
            else:
                print('Some correlation (reject H0)')
        else:
            print(x1.reshape(-1), x2.reshape(-1))
            print("Data not valid")

    @classmethod
    def comp_ttest(cls, x1, x2, ind=True):
        """
        Description:
            Compute t-test distribution (independent and related) between two data samples x1, x2
        :param x1: numpy array 
        :param x2: numpy array
        :ind: independent parameter (default = True)
        """
        if ind:
            # independent student's t-test
            stat, p = ttest_ind(x1, x2)
            print('Statistics=%.3f, p=%.3f' % (stat, p))
        else:
            # Paired student's t-test
            stat, p = ttest_rel(x1, x2)
            print('Statistics=%.3f, p=%.3f' % (stat, p))
        
        # Interpret p-value
        alpha = 0.05
        if p > alpha:
            print('Same distributions (fail to reject H0)')
        else:
            print('Different distributions (reject H0)')
        
        return stat, p


    @classmethod
    def gen_corr_mx(cls, data):
        """
        Description: generate correlation matrix, reduce the Dimension using Truncated SVD Singular Value Decomposition

        References:
            https://en.wikipedia.org/wiki/Singular-value_decomposition
            https://medium.com/@jonathan_hui/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491
            http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.corrcoef.html

        :param data: pandas.core.frame.DataFrame
        :return: Truncated SVD object
        """
        print(data.corr())
        svd = TruncatedSVD(n_components=2, n_iter=5, random_state=None)
        return svd.fit(data.values)

    @classmethod
    def comp_cov_mx(cls, data1, data2=None):
        """
        Description: compute covariance matrix between two variables for computing the PCA
        Sigma = 1/m*data_transposed*data with m is number of training data
        """
        if data2:        
            return cov(data1, data2)
        else:
            if isinstance(data1, np.ndarray):
                return cov(data1.T)
            else:
                return

    @classmethod
    def detect_outliers(cls, data):
        """
        Description: 
            Algorithm to detect outlier            
            Choose to remove (clean) or neutralize using Minkowski-method

            Algorithm functions in combination with plot-visual and observation.
            Outliers are the points outside the range [(Q1-1.5 IQR), (Q3+1.5 IQR)]

        References:
            https://www.neuraldesigner.com/blog/3_methods_to_deal_with_outliers
            http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm

        :param data: pandas.core.series.Series
        :return: list of all outliers 
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

            # find max value
            max_value = data[col].describe()['max']

            IQR = abs(q3_upper_quartile - q1_lower_quartile)
            IQR = 1.5*IQR
            lower_bound = q1_lower_quartile - IQR
            upper_bound = q3_upper_quartile + IQR

            # print(min_value, q1_lower_quartile, q2_median, q3_upper_quartile, max_value, lower_bound, upper_bound)
            outliers = [v for v in data[col].values if v < lower_bound or v > upper_bound]

            # print(col, outliers, len(outliers))
            total_outliers[col] = outliers

        # tbd Step2: find all row_idx of these outliers points which appear in all columns
        #
        # then find the occurrences > 2
        #
        # tbd Step3: remove or keep it by modifying its values
        return total_outliers

    @classmethod
    def detect_density_anomaly(cls, data, feature_column, eps):
        """
        Description: detect anomaly features using K-NN and relative density of data
            setup a threshold epsilon for error (maximal error).
            if all data points exceeds this threshold, then we list them all as outlier/ anomaly.
            Application like: Fraud detection, detecting abnormal or unusual observations
            Method: using simple moving average (SMA) or low-pass filter

            Transform non_Gaussian features into Gaussian features using e.g. function log(..)
            Using:
                1. Original model
                    manually create additional features to capture anomalies (unusual combinations of values)
                2. Multivariate Gaussian Distribution,
                    calculate population mean, variance matrix Sigma
                    automatically capture correlations between features
                3. Manually create additional features to help capturing the unusual anomaly combination of values
                    (Requirement: m_training set > n_number_of_features, e.g. m = 10.n)                

        References:
            Hui Xiong, Gaurav Pandey, Michael Steinbach, Vipin Kumar (Fellow, IEEE): Enhancing Data Analysis with Noise Removal
            https://en.wikipedia.org/wiki/Anomaly_detection
            https://www.datascience.com/blog/python-anomaly-detection            
            
        :param data: pandas.core.series.Series
        :param feature_column: column value where need to be converted to log-values
        :param eps: epsilon as threshold where abnormal data are being detected
        :return:
        """
        data['log_val'] = np.log(data[feature_column].values)
        pass

    @classmethod
    def detect_cluster_anomaly(cls, data, feature_column, eps):
        """
        References:
            https://www.datascience.com/blog/python-anomaly-detection
        """
        pass

    @classmethod
    def detect_svm_anomaly(cls, data, feature_column, eps):
        """
        References:
            https://www.datascience.com/blog/python-anomaly-detection
        """
        pass
        
    @classmethod
    def applyby_func(cls, data, columns, ops):
        """
        Description: apply func on certain columns (to change or update values)

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
        Description:
            filter/ drop out all the values (rows) which are considered "No need" for dataset
            to reduce the unnecessary data processing (using DataFrame_apply)

        :param data:
        :param func:
        :return:
        """
        return data.drop(data[func].index)

    @classmethod
    def unionby_rows(cls, data, func):
        """
        Description: union all rows together with the pre-defined values (using DataFrame_apply)

        :param data:
        :param func:
        :return:
        """
        # data.apply(func, )
        pass

    @classmethod
    def mergeby_dataset(cls, data1, data2):
        """
        Description: union data1 and data2 and filter out all duplicates (using DataFrame_merge)

        :param data1: first data package
        :param data2: second data package
        :return:
        """
        pass

    @classmethod
    def joinby_dataset(cls, data1, data2):
        """
        Description: proceed different join_operations (left, right, inner, outer) on dataset (using DataFrame_merge)
            (similar as mergeby_data)

        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def intersectby_dataset(cls, data1, data2):
        """
        Description: proceed operation intersect to get the common part between data1 and data2 (using DataFrame_merge)

        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def mergeby_data(cls, data, rows, columns):
        """
        Description: merge values from rows and columns together

        :param data:
        :param rows:
        :param columns:
        :return:
        """
        pass

    @classmethod
    def countby_values(cls, data, value=None):
        """
        Description: count value by key (using DataFrame_to_dict to convert DataFrame into dict_type)

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
        Description: 
            sort (ascending, descending) of rows, columns (using DataFrame.sort_values)

        :param data:
        :param keyvalue:
        :return:
        """
        pass

    @classmethod
    def lookup_value(cls, data, value, tup=True):
        """
        Description: 
            find all values in data frame

        :param data: pandas.core.frame.DataFrame
        :param value: type either int, float or object
        :param tup: if True, it returns as tuple with column. Otherwise, it will return a list of row_id
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
    def searchByValue(cls, data, try_keys, value):
        """
        Description:
            filter out data from certain column with specific value
            (e.g. in case there are many different key_columns names
        :param data: pandas.core.frame.DataFrame
        :param try_keys: list of all possible key_columns to search 
        :param value: value in column need to be filtered
        """
        # return data[data[column]==value]
        i=0
        while (i<len(try_keys)):
            if try_keys[i] in data:
                # TODO: use str.extract to suppress the UserWarning                
                return data[data[try_keys[i]].str.contains(value)]                
            else:        
                i=i+1

        # return data[data[try_keys].str.contains(value)]
        print("Nothing is found")
        return

    @classmethod
    def searchByValueColumn(cls, data, try_keys, column, value):
        """
        Description:
            filter out data from certain column with specific value
        :param data: pandas.core.frame.DataFrame
        :param try_keys: list of all possible key_columns to search 
        :param column: identify data from certain column
        :param value: value in column need to be filtered
        """
        # return data[data[column]==value]
        i=0
        while (i<len(try_keys)):
            if try_keys[i] in data:
                return data[data[column]>0][data[data[column]>0][try_keys[i]] == value]
                # return data[data[column]>0].groupby(by=try_keys[i]).sum().loc[value][column]
            else:        
                i=i+1

        # return data[data[try_keys].str.contains(value)]
        print("Nothing is found")
        return
        

    # @classmethod
    # def searchByValue2(cls, data, column, value):
    #     """
    #     Description:
    #         filter out data from certain column with specific value

    #     tbd.
    #     """
    #     pattern = '*'+value+'*'
    #     filtered = fnmatch.filter(data, pattern)  
        
    #     return data[data[column].str.contains(value)]
        

    @classmethod
    def mean_neighbors(cls, data, row_id, column):
        """
        Description:
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
        Description: 
            return name of all columns which have NaN_value

        :param data: pandas.core.frame.DataFrame
        :return: list of all possible NaN_column(s)
        """
        # others: return [col for col in data.columns if data[col].isnull().sum() > 0] 
        nan_bool = data.isnull().any()
        # if isinstance(data, dask.dataframe.core.DataFrame):
        if isinstance(data, pd.core.frame.DataFrame):
            key_test = [key for key, value in nan_bool.iteritems() if value]
            return key_test
        else:
            return nan_bool

    @classmethod
    def nan_rows(cls, data, nan=True):
        """
        Description: 
            return all rows containing NaN values in type DataFrame

        :param data: pandas.core.frame.DataFrame
        :param nan: Boolean-input True to search for NaN values, False for not_NaN
        :return: data with all possible found NaN_rows or not_NaN_rows (if nan=False)
        """
        if isinstance(data, dask.dataframe.core.DataFrame) and nan:
            return data[data.isnull().any(axis=1)]

        elif isinstance(data, dask.dataframe.core.DataFrame) and not nan:
            return data[data.notnull().any(axis=1)].dropna(axis=0, how='any')

        else:
            return data.isnull().any()

    @classmethod
    def pop_rows(cls, data, idx, inplace=True):
        """
        Description: 
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
        Description: 
            pre_processing columns based on information given in the config.ini

        :param data: pandas.core.frame.DataFrame
        :return: data after pre-processing
        """
        # find columns containing NaN-value
        nan_cols = cls.nan_columns(data)
                       
        # if nan_cols and Start.kwargs['drop_obj_col']:            
        #     # drop all columns with type object if all values are NA, drop that label
        #     data = data.dropna(how='all')
        #     # return data.select_dtypes(exclude=object)
        #     return data

        if nan_cols and Start.kwargs['nan_drop_col']:
            # drop all nan_columns, axis : {0 or 'index (rows)', 1 or 'columns'}
            print("-> Drop all columns: ", nan_cols)   
            return data.drop(nan_cols, axis=1)

        elif nan_cols and Start.kwargs['nan_zero']:
            # convert nan_value in column into zero_value (WARNING: columns in dtypes float64, int64), axis=1
            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    print("-> Replace NaN value by value 0 at column: ", nan_col)                    
                    data[nan_col] = data[nan_col].mask(data == np.NaN, 0)
            return data

        elif nan_cols and Start.kwargs['nan_mean']:
            # convert nan into mean_value of column (WARNING: only suitable for columns in dtypes float64, int64)
            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    print("-> Replace NaN value by mean-value at column: ", nan_col)                    
                    data[nan_col] = data[nan_col].mask(np.isnan(data[nan_col]), np.round_(data[nan_col].mean().compute(), 2))
                    print("-> Replace NaN value by mean-value at column: RESULT SETUP")

                elif data[nan_col].dtype == np.object:
                    print("-> Drop column: ", nan_col)
                    data = data.drop(nan_col, axis=1)
            return data

        else:
            return data

    @classmethod
    def process_nan_rows(cls, data):
        """
        Description: 
            pre_processing rows based on information given in the config.ini

        :param data: pandas.core.frame.DataFrame
        :return: data after pre-processing
        """
        # tbd: improve performance

        nan_cols = cls.nan_columns(data)
        nan_rows = cls.nan_rows(data)

        # drop the duplicates rows
        data = data.drop_duplicates()

        if not nan_rows.empty and Start.kwargs['nan_drop_row']:
            # Drop the rows where all elements are nan
            data = data.dropna(axis=0, how='all')

            # Drop row if it does not have at least two values that are **not** NaN
            return data.dropna(thresh=2)

        elif not nan_rows.empty and Start.kwargs['nan_zero']:

            # convert nan in row into zero_value, axis=0
            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)
            return data

        elif not nan_rows.empty and Start.kwargs['nan_mean']:

            for nan_col in nan_cols:
                if data[nan_col].dtype == np.float64 or data[nan_col].dtype == np.int64:
                    # data[nan_col] = data[nan_col].replace(to_replace=np.NaN, value=0)

                    # compute the mean of neighbor-values, option: 'most_frequent', 'median'
                    imputer = SimpleImputer(missing_values='NaN', strategy='mean', axis=1)
                    imputer = imputer.fit(data[nan_col].values.reshape(1, -1))
                    data[nan_col] = imputer.transform(data[nan_col].values.reshape(1, -1))[0]
            return data

        elif not nan_rows.empty and Start.kwargs['nan_mean_neighbors']:

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
        Description: 
            simply process all nan-value filled by 'Unknown'

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
        Description: 
            merge dataframes by applying the common features in between data frames

        :param data: list of data frames
        :param feature: the common feature
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
        Description: 
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
        Description:
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
        Description:    
            display summary about all NaN values in data

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        print("Nans_columns: \n{}".format(StartML.nan_columns(data)))
        print("Nans_rows: \n{}".format(len(StartML.nan_rows(data))))

    @staticmethod
    def fiveNumber_summary(data):
        """
        Description: 
            The five-Number summary describes data sample with any distribution
        Reference: 
            https://en.wikipedia.org/wiki/Five-number_summary
        """
        # calculate quartiles from 25, 50 (median), 75
        quartiles = percentile(data, [25, 50, 75])

        # calculate min/max
        data_min, data_max = data.min(), data.max()

        # display 5-number summary
        print('Min: %.3f' % data_min)
        print('Q1: %.3f' % quartiles[0])
        print('Median: %.3f' % quartiles[1])
        print('Q3: %.3f' % quartiles[2])
        print('Max: %.3f' % data_max)

    @staticmethod
    def summary(data):
        """
        Description: 
            Show all basic information about data set
        
        :param data: pandas.core.frame.DataFrame
        """
        print("\Data type: {}".format(type(data)), "\n")
        print("Data Columns: {}".format(data.columns), "\n")
        print("Missing values in Data: \n{}".format(data.isnull().sum()), "\n")
        print("Raw Data first look at data: \n{}".format(data.head(10)), "\n")
        print("Dimension of Data: \n{}".format(data.shape), "\n")
        print("Data Type for all Attributes/ features: \n{}".format(data.dtypes), "\n")
        print("Data Information: \n{}".format(data.info()), "\n")
        print("Descriptive Statistics (Count, Mean, Standard Deviation, Minimum-, (25, 50, 75) Percentile, -Maximum): \n{}".format(data.describe()), "\n")        
        print(StartML.nan_summary(data))

    @staticmethod
    def pipeline_run():
        """
        Description: 
            run data processing pipeline
        """
        pass
