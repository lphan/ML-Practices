import configparser
import json 
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np

from dask.delayed import delayed

from pickle import dump
from pickle import load

from sklearn.externals.joblib import dump as joblib_dump
from sklearn.externals.joblib import load as joblib_load
from sklearn.utils import shuffle

from pip._internal.operations.freeze import freeze


class Start(object):
    """
    Description: StartML - Start Machine Learning
    Import parameters from config.ini and execute basic pre_processing operations
    (statistics, groupby, reduce, etc.) on datasets

    Start: 
        jupyter notebook
        -> from start import *
        -> info_help
    """
    # init keywords arguments
    kwargs = {}

    def __init__(self):
        pass

    @staticmethod
    def _arguments():
        """
        Description: read config-parameters from local file config.ini
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

        exclude_obj_col = config.getboolean('Start', 'exclude_object_column')
        nan_drop_col = config.getboolean('Start', 'nan_drop_column')
        nan_drop_row = config.getboolean('Start', 'nan_drop_row')
        nan_zero = config.getboolean('Start', 'nan_zero')
        nan_mean = config.getboolean('Start', 'nan_mean')
        nan_mean_neighbors = config.getboolean('Start', 'nan_mean_neighbors')

        Start.kwargs.update({"data_path": data_path,
                               "drop_obj_col": exclude_obj_col,
                               "nan_drop_col": nan_drop_col,
                               "nan_drop_row": nan_drop_row,
                               "nan_zero": nan_zero,
                               "nan_mean": nan_mean,
                               "nan_mean_neighbors": nan_mean_neighbors})
        print("local_kwargs", Start.kwargs)
    
    @classmethod
    def saveModel(cls, model, filename, joblib=False):
        # Find and export the Python version and Library version 
        lib = [requirement for requirement in freeze(local_only=True)]
        dependencies = dict([(li.split('==')[0], li.split('==')[1]) for li in lib])

        with open('dependencies.json', 'w') as outfile:
            json.dump(dependencies, outfile)

        if joblib:
            # serialize with Numpy arrays by using library joblib
            joblib_dump(model, filename)
        else:
            # serialize standard Python objects
            dump(model, open(filename, 'wb'))

    @classmethod
    def loadModel(cls, model, dependencies_filename, joblib=False):
        dependencies = json.load(dependencies_filename)

        if joblib:
            # deserialize by using library joblib
            loadedmodel = joblib_load(model)
        else:
            # deserialize standard Python objects
            loadedmodel = load(open(model, 'rb'))
        
        return loadedmodel, dependencies


    @staticmethod
    def import_data():
        """
        Description: 
            read data from data_set .csv and convert them into Pandas Data Frame and append them into a list

        References:
            https://docs.python.org/3/library/codecs.html#standard-encodings
        """
        Start._arguments()
        if not Start.kwargs:
            return

        df = []
        try:
            if Start.kwargs['data_path']:
                paths = Start.kwargs['data_path'].split(',')
                for path in paths:

                    # remove space before and after string
                    path = path.strip()

                    # Delimiter processing
                    if path.endswith('.xlsx') or path.endswith('.xls'):
                        # data_exl = pd.read_excel(path)
                        parts = dask.delayed(pd.read_excel)(path)
                        data_exl = dd.from_delayed(parts)
                        df.append(data_exl)
                    elif path.endswith('.json'):
                        # data_json = pd.read_json(path)
                        data_json = dd.read_json(path)
                        df.append(data_json)
                    elif path.endswith('.csv'):
                        # data_csv = pd.read_csv(path, low_memory=False)
                        data_csv = dd.read_csv(path)
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

# Pre-Initialized the object idata    
idata = []
for dat in Start.import_data():
    # if isinstance(dat, pd.DataFrame):
    if isinstance(dat, dd.DataFrame):
        dat.columns = [col.strip() for col in dat.columns]
        idata.append(dat)