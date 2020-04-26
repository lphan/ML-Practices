import configparser
import json 
import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os

from dask.delayed import delayed

from pickle import dump
from pickle import load

from sklearn.externals.joblib import dump as joblib_dump
from sklearn.externals.joblib import load as joblib_load
from sklearn.utils import shuffle

from pip._internal.operations.freeze import freeze
from keras.models import model_from_json
from keras.models import model_from_yaml


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
        folder_path = config['paths']['folder_path']
        data_path = config['paths']['data_path']

        exclude_obj_col = config.getboolean('Start', 'exclude_object_column')
        nan_drop_col = config.getboolean('Start', 'nan_drop_column')
        nan_drop_row = config.getboolean('Start', 'nan_drop_row')
        nan_zero = config.getboolean('Start', 'nan_zero')
        nan_mean = config.getboolean('Start', 'nan_mean')
        nan_mean_neighbors = config.getboolean('Start', 'nan_mean_neighbors')
        pandas_type = config.getboolean('Start', 'pandas_type')

        Start.kwargs.update({"folder_path": folder_path,
                               "data_path": data_path,
                               "drop_obj_col": exclude_obj_col,
                               "nan_drop_col": nan_drop_col,
                               "nan_drop_row": nan_drop_row,
                               "nan_zero": nan_zero,
                               "nan_mean": nan_mean,
                               "nan_mean_neighbors": nan_mean_neighbors, 
                               "pandas_type": pandas_type})
            
    @classmethod
    def saveDependencies(cls, model, filename, joblib=False):
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
    def serializeModels(cls, model, type):
        # load json
        if type=='json': 
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
        elif type=='yaml':
            model_yaml = model.to_yaml()
            with open("model.yaml", "w") as yaml_file:
                yaml_file.write(model_yaml)
            # serialize weights to HDF5
            model.save_weights("model.h5")
        else:
            print("Data type is invalid")

    @classmethod
    def loadModels(cls, filename, type):
        # load json and create model
        if type=='json':
            json_file = open(filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
        elif type=='yaml':
            yaml_file = open(filename, 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            loaded_model = model_from_yaml(loaded_model_yaml)

        return loaded_model

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
    def import_data(path):
        """
        Description: 
            read data from data_set .csv and convert them into Pandas/ Dask Data Frame and append them into a list

        References:
            https://examples.dask.org/dataframes/01-data-access.html
        """
        # input configuration parameters  
        # Start._arguments()
        # if not Start.kwargs:
        #     return

        pandas = Start.kwargs['pandas_type']

        try:
            # Delimiter processing
            if path.endswith('.xlsx') or path.endswith('.xls'):
                if pandas:
                    df = pd.read_excel(path)
                else:
                    parts = dask.delayed(pd.read_excel)(path)
                    df = dd.from_delayed(parts)
        
            elif path.endswith('.json'):
                if pandas:
                    df = pd.read_json(path)
                else:
                    df = dd.read_json(path)
                
            elif path.endswith('.csv'):
                if pandas:
                    df = pd.read_csv(path, low_memory=False)
                else:
                    df = dd.read_csv(path)
                
            else:
                # print('Unknown format')
                return
        
        except (TypeError, OSError, FileNotFoundError):
            print("Wrong Type Format of imported data")
            import sys
            sys.exit(1)

        return df

    @staticmethod
    def import_folder():
        # input configuration parameters  
        Start._arguments()        
        folder = Start.kwargs['folder_path']

        if not folder:
            return
    
        files = os.listdir(folder)
        idata = []
        print(files)
        for fil in files:
            pathfile = folder + fil            
            df = Start.import_data(pathfile)
            # df.columns = [col.strip() for col in df.columns]
            if df is not None:
                idata.append(df)
            else:
                print("None")

        return idata          

Start._arguments()

if Start.kwargs['folder_path']:
    print("Start importing files in folder")   
    data = Start.import_folder()

if Start.kwargs['data_path']:
    filename = Start.kwargs['data_path']
    print("Start importing single data ", filename)
    sdata = Start.import_data(filename)
else: 
    print("No Data_Path or Folder_Path is given")
    
# Persist data in memory to allow future computations faster (only apply for dask-object)
if Start.kwargs['pandas_type'] is False:
    data = [dat.persist() for dat in data]
