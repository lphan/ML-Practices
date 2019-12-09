import json 

from pickle import dump
from pickle import load

from sklearn.externals.joblib import dump as joblib_dump
from sklearn.externals.joblib import load as joblib_load

from pip._internal.operations.freeze import freeze

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