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


# import configparser
# import pandas as pd
# import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row

import collections

class StartSPK(object):
    """
    Description: StartSPK - Start Apache Spark
    Operations for large-scale on datasets

    References:
        https://pages.databricks.com/mastering-advanced-analytics-apache-spark.html
        https://spark.apache.org/sql/
        https://spark.apache.org/docs/latest/sql-programming-guide.html

    Start:
        jupyter notebook
        -> from startspk import *
        -> info_help
    """
    # init keywords arguments
    kwargs = {}

    def __init__(self, app_name, path_file, config_opt="", config_val="", rdd=True):

        # create SparkContext for RDD object and import data from source
        # self.spark_conf = (SparkConf().setAppName(appname))
        self.spark_conf = SparkConf().setMaster("local").setAppName(app_name).set(config_opt, config_val)
        self.spark_cont = SparkContext(conf=self.spark_conf)

        # create SparkSession for DataFrame object
        self.spark_sess = (SparkSession.builder.appName(app_name)
                           .config(config_opt, config_val).getOrCreate())

        if rdd:
            self.data = self.spark_cont.textFile(path_file)
        else:
            self.data = self.spark_sess.sparkContext.textFile(path_file)

    def get_dat(self):
        return self.data, self.spark_cont, self.spark_sess

    @classmethod
    def create_SpkObj(cls, data):
        """

        :param data:
        :return:
        """
        pass

    @classmethod
    def execute_SpkOps(cls, data):
        """

        :param data:
        :return:
        """
        pass

    @staticmethod
    def info_help():

        return {
            "info_help_StartSPK": StartSPK.__name__,
            "StartSPK.kwargs": "Show key words arguments from config.ini",
        }


class SparkRDD(StartSPK):

    def __init__(self, app_name, path_file):
        StartSPK.__init__(self, app_name, path_file)

    @classmethod
    def execute_RDDops(cls, data):
        """

        :param data: DataSet
        :return:
        """
        pass

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

        :param data: pandas.core.frame.DataFrame
        :param columns: list of columns need to be grouped
        :param groupby_label: need to be one of the given columns
        :param func:
        :return: DataFrameGroupBy object (which can be used to compute further)
        """
        grouped = data[columns].groupby(groupby_label)
        if func is None:
            return grouped
        else:
            return grouped.aggregate(func)

    @classmethod
    def reduceby_rows(cls, data, operations):
        """
        reduce number of rows (map_reduce)

        :param data:
        :param rows:
        :param operation: add, subtract, multiplication, mean etc.
        :return:
        """
        pass

    @classmethod
    def filterby_rows(cls, data, values):
        """
        filter out all the values (rows) which are considered "No need" for dataset
        to reduce the unnecessary data processing

        :param data:
        :param values:
        :return:
        """
        pass

    @classmethod
    def unionby_rows(cls, data, values):
        """
        union all rows together with the pre-defined values
        
        :param data:
        :param values:
        :return:
        """
        pass

    @classmethod
    def unionby_data(cls, data1, data2):
        """
        union data1 and data2 and filter out all duplicates

        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def joinby_data(cls, data1, data2):
        """
        proceed different join_operations (left, right, inner, outer) on dataset

        :param data1:
        :param data2:
        :return:
        """
        pass

    @classmethod
    def intersectionby_data(cls, data1, data2):
        """
        proceed operation intersect to get the common part between data1 and data2

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
    def countby_kv(cls, data, keyvalue, operations):
        """
        count value with key

        :param data:
        :param keyvalue:
        :param operations: (e.g. sum)
        :return:
        """
        pass

    @classmethod
    def orderby_kv(cls, data, keyvalue):
        """
        sort (ascending, descending) of rows, columns

        :param data:
        :param keyvalue:
        :return:
        """
        pass


class StartSpkSQL(StartSPK):

    def __init__(self, app_name, path_file):
        StartSPK.__init__(self, app_name, path_file)

    def map_data(self, data):
        """
        Map and create dataframe object
        :param data:
        :return:
        """
        pass

    @classmethod
    def sparksql(cls,data):
        """
        process all SQL-spark style with dataframe

        :param data: DataFrame
        :return:
        """
        pass

    @classmethod
    def query_SparkSQL(cls, data, sql_command):
        """

        :param data:
        :return:
        """
        # scheme = spark.createDataFrame(data)
        # scheme.createOrReplaceTempView("title")
        # sql_obj = spark.sql(sql_command)
        pass

    @classmethod
    def execute_SQLops(cls, data):
        """

        :param data: DataFrame
        :return:
        """
        pass


# E.g. data = StartSPK(appname='AppName', pathfile='.../file.csv')
info_ml = StartSPK.info_help()
StartSPK.__doc__
