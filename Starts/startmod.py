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


import dask
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from Starts.startml import *
from Starts.startvis import *
from scipy.stats import uniform
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.utils import resample

# from sklearn.metrics import huber
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from scipy.stats import kurtosis, skew
from math import sqrt
# from sklearn.pipeline import make_pipeline
# from sklearn.ensemble.partial_dependence import plot_partial_dependence
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency
from scipy.stats import chi2

class StartMod(StartML):
    """
        Description: StartMod - Start Models
        Apply Machine Learning Models in: 
            Regression: Linear, Multivariants, Logistics
            Classification k-NN, Decision Tree, Naiv Bayes, SVM, Neural Network.

        Start:
            jupyter notebook
            -> from startmod import *
            -> info_mod
    """

    def __init__(self, n_classes, dependent_label):
        """
        setup parameters for model
        :param n_classes: number of classes as the value of dependent_label (e.g n_classes=2 as binary classification)
        :param dependent_label: target_feature
        """

        if dependent_label:
            self.n_classes = n_classes
            self.dependent_label = dependent_label

        else:
            self.n_classes = n_classes
            self.dependent_label = None  # there is no dependent_label (e.g. unsupervised learning)

    def _get_attributes(self):
        return {'dependent_label': self.dependent_label, 'n_classes': self.n_classes}

    def _set_attributes(self, dict_params):
        self.n_classes = dict_params['n_classes']
        self.dependent_label = dict_params['dependent_label']

    update_parameters = property(_get_attributes, _set_attributes)

    @classmethod
    def encode_label_column(cls, data, label_columns, one_hot=False):
        """
        Description: encode object-columns
            This encoding is needed for feeding categorical data to many scikit-learn estimators,
            notably linear models and SVMs with the standard kernels.

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

        :param data: pandas.core.frame.DataFrame
        :param label_columns: list of all labels (object-type)
        :param one_hot: Boolean-value, set value True for categorical column
        :return: data and x_values (the encoded data in type numpy.array)
        """

        for label_col in label_columns:            
            # Encode label only applies to Column in type Object
            if data[label_col].dtype is not np.object:
                print("Type of label_column " + label_col + " is " + str(data[label_col].dtype))
            
            return data

        x_values = data.values
        try:
            for label_col in label_columns:
                label_idx = data.columns.get_loc(label_col)

                # label_encoder to turn object-column into number-column
                label_encoder = LabelEncoder()
                x_values[:, label_idx] = label_encoder.fit_transform(x_values[:, label_idx])
        except TypeError:
            print("Data might have mixed type str and float, please make sure there is no (float) NaN-Value")

        # one_hot=True is used for categorical column
        if one_hot:
            try:
                labels_idx = []
                for label_col in label_columns:
                    # get column index
                    label_idx = data.columns.get_loc(label_col)
                    # data.values[:, label_idx] = label_encoder.fit_transform(data.values[:, label_idx])
                    labels_idx.append(label_idx)

                # create dummy columns to encode the above label_column
                one_hot_encoder = OneHotEncoder(categorical_features=labels_idx)

                # data.values[:, label_idx]
                # = one_hot_encoder.fit_transform(data.values[:,label_idx].reshape(1,-1)).toarray()
                x_values = one_hot_encoder.fit_transform(x_values).toarray()

                # remove dummy trap
                # x_values = x_values[:, 1:]

                # TODO: replace data.columns into new_data.columns (parameter to transfer column name in fit_transform?)
                # new_data = pd.DataFrame(data=new_values, columns=...)
                data = pd.DataFrame(data=x_values)

                # convert to numeric-type DataFrame
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col])
            except ValueError:
                print("Data might be containing NaN_values")

            return data

        else:
            # one_hot=False is used for object_feature columns
            data = pd.DataFrame(data=x_values, columns=data.columns, index=None)
            # convert to numeric-type DataFrame
            for col in data.columns:
                data[col] = pd.to_numeric(data[col])
            return data

    @classmethod
    def split_columns(cls, data, cols):
        """
        Description: split data by feature_columns into 2 different datasets

        :param data: pandas.core.frame.DataFrame
        :param cols: list of columns feature e.g. cols = [['a', 'b'], 'c']
        :return: list of pandas data frames splitted by columns
        """
        dat = [data[cols[i]] for i in range(len(cols))]
        return dat

    @classmethod
    def split_data(cls, data, dependent_label=None, t_size=0.2, seed=0, type_pd=True, split=True):
        """
        Description:
            split data by rows into training_data and test_data used for (regression, classification) methods

        References:
            https://machinelearningmastery.com/difference-test-validation-datasets/
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
            
        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical label
        :param t_size: test_size (default is 0.2)
        :param seed: random state seed (default is 0)
        :param type_pd: (default is Pandas Dataframe)
        :param split: (default is True)
        :return: x_train, x_test, y_train, y_true (default type Pandas DataFrame)
        """

        # convert data into numpy-values (in case: the last column is dependent label)
        # x = data.iloc[:, :-1].values
        # y = data.iloc[:, 1].values
        # save the dependent value into y

        if not dependent_label and not type_pd and isinstance(data.values, np.ndarray):
            # split data into train and test in ratio 8:2
            train, test = train_test_split(data, test_size=t_size)
            return train, test

        if type_pd:
            # keep type Pandas DataFrame
            y = data[dependent_label]
            x = data.drop([dependent_label], axis=1)
        else:
            # convert to type Numpy
            y = data[dependent_label].values
            # drop dependent value from data and save the independent values into x
            x = data.drop(dependent_label, axis=1).values

        if not split:
            return x, y

        try:
            # split data into training set and test set
            x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=t_size,
                                                                random_state=seed, shuffle=True)

        except ValueError:
            print("Data set is not valid yet, need to be preprocessed first, No splitting happen")
            return data

        return x_train, x_test, y_train, y_true

    @classmethod
    def split_data_validate(cls, data, dependent_label=None, test_size=0.2, random_state=0, type_pd=True, split=True, cv=False):
        '''
        Description:
            split data by rows into training_data, validation_data and test_data
            Default folding for cross validation: k_fold = 10

        References:
            https://machinelearningmastery.com/difference-test-validation-datasets/

        :param data: pandas.core.frame.DataFrame
        :param cv: cross_validation to split data into 3 training, validation and test sets (default is False)
        :param dependent_label: categorical label
        :param test_size: (default is 0.2)
        :param random_state: (default is 0)
        :param type_pd: (default is Pandas Dataframe)
        :param split: (default is True)
        '''
        if not dependent_label and not type_pd and isinstance(data, np.ndarray):
            # split data into train and test in ratio 8:2
            train, test = train_test_split(data, test_size=test_size)
            return train, test

        if type_pd:
            # keep type Pandas DataFrame
            y = data[dependent_label]
            x = data.drop([dependent_label], axis=1)
        else:
            # convert to type Numpy
            y = data[dependent_label].values
            # drop dependent value from data and save the independent values into x
            x = data.drop(dependent_label, axis=1).values

        # split data into training set, validation set and test set
        x_data, x_test, y_data, y_true = train_test_split(x, y, test_size=test_size,
                                                          random_state=random_state, shuffle=True)

        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=test_size,
                                                          random_state=random_state, shuffle=True)    

        # tune model hyperparameters        
        # if parameters:
        #     for params in parameters:
        #         skills = list()
        #         for i in k_fold:
        #             fold_train, fold_validation = train_test_split(i, k_fold, train)
        #             model = fit(fold_train, params)
        #             skill_estimate = evaluate(model, fold_validation)
        #             skills.append(skill_estimate)
        #         skill = summarize(skills)
        
        return x_train, x_val, x_test, y_train, y_val, y_true

    @classmethod
    def backward_eliminate(cls, data, x_data, y_data):
        """
        Description:
            support the evaluation on (regression) models by finding maximal p_value (< pre-defined SL)
            and applying method Backward Elimination for feature selection

        References:
            http://www.stephacking.com/multivariate-linear-regression-python-step-6-backward-elimination/

        :param data: pandas.core.frame.DataFrame
        :param x_data: feature_values in type numpy.ndarray
        :param y_data: categorical_values in numpy.ndarray
        :return: regressor_ols object, max_pvalue, x_opt
        """

        # Start Method Backward Elimination for feature selection
        # Step 1: init a significance level to stay
        sl = 0.05

        # initiate data for x with full columns
        x_opt = x_data[:, [i for i in range(len(data.columns))]]

        # Step 2: fit the full model with all possible predictors, create new object in (Ordinary Least Squares) OLS
        reg_ols = sm.OLS(endog=y_data, exog=x_opt).fit()

        # Step 3: select and remove the redundant columns with pvalue > significant level 0.05
        max_pvalue, col_idx = StartML.find_idx_max_value(reg_ols.pvalues)

        while max_pvalue > sl:
            # remove value at column col_idx and refresh data of x_opt
            x_opt = np.delete(x_opt, col_idx, axis=1)

            # recompute regressor with new x_opt
            reg_ols = sm.OLS(endog=y_data, exog=x_opt).fit()
            max_pvalue, idx = StartML.find_idx_max_value(reg_ols.pvalues)

        return reg_ols, max_pvalue, x_opt

    @classmethod
    def feature_columns(cls, data, label=None):
        """
        Description: find and return object and non-object columns
        
        :param data: pandas.core.frame.DataFrame
        :param label: default is None
        :return: list of non_obj_feature, list of obj_feature
        """
        if label is None:
            non_obj_feature = [lab for lab in data.columns if data[lab].dtype != 'object']
            obj_feature = [lab for lab in data.columns if data[lab].dtype == 'object']
            return non_obj_feature, obj_feature
        else:
            # x, y = train_data, train_data.pop('Time') will remove column 'Time' completely out of train_data
            train_x, _ = StartMod.split_data(data, split=False)
            non_obj_feature = [lab for lab in train_x.columns if train_x[lab].dtype != 'object']
            obj_feature = [lab for lab in train_x.columns if train_x[lab].dtype == 'object']

            return non_obj_feature, obj_feature

    @classmethod
    def feature_scaling(cls, data, scale='standard', feature_range=None):
        """
        Description: 
            - Minmax Rescaling is to normalize data value into the feature range between Min 0 and Max 1
            - Standardization involves rescaling the features such that they have the properties
            of a standard normal distribution with a mean of 0 and a standard deviation of 1

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

        :param data: pandas.core.frame.DataFrame or dask.dataframe.core.DataFrame or numpy.array
        :param feature_range: default (0,1)
        :return: data in scaled format
        """
        if isinstance(data, dask.dataframe.core.DataFrame):
            # convert data in Pandas DataFrame, apply Min_Max method manually
            # data[data.columns] = data[data.columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
            print(data.dtype)
            func = lambda x: (x - x.min()) / (x.max() - x.min())
            return data.apply(func, axis=1)

        elif isinstance(data, pd.core.frame.DataFrame): 
            print(type(data))
            data[data.columns] = data[data.columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)            
            return data

        else:
            if scale is 'standard':
                scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
                
                array = data.values
                X = array[:, 0:len(array)]
                Y = array[:, len(array)]     
                rescaled_X = scaler.fit_transform(X)                
                    
                np.set_printoptions(precision=3)
                return rescaled_X, Y  

            elif scale is 'minmax':
                if not feature_range:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                else:
                    scaler = MinMaxScaler(feature_range=feature_range)
            
                array = data.values
                X = array[:, 0:len(array)]
                Y = array[:, len(array)]     
                rescaled_X = scaler.fit_transform(X)                
                    
                np.set_printoptions(precision=3)
                return rescaled_X, Y 

            elif scale is 'binary':
                array = data.values
                X = array[:, 0:len(array)]
                Y = array[:, len(array)]     
                binarized = Binarizer(threshold=0.0).fit(X)
                binary_X = binarized.transform(X)
                np.set_printoptions(precision=3)
                return binary_X, Y

            else:    
                array = data.values
                X = array[:, 0:len(array)]
                Y = array[:, len(array)]

                scaler = Normalizer().fit(X)                    
                normalized_X = scaler.transform(X)

                np.set_printoptions(precision=3)
                return normalized_X, Y

    @classmethod
    def feature_selection(cls, data, rm_columns, dependent_label=None, rm=False, pr=True):
        """
        Description:
            function to simplify feature selection and dimensionality reduction respectively
            apply Backward Elimination, Forward Selection, Bidirectional Elimination, Score comparision

        References:
            http://scikit-learn.org/stable/modules/feature_selection.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: label of categorical column
        :param rm_columns: list of feature_columns which will be removed
        :param rm: default False (if True: columns from rm_columns will be removed)
        :param pr: turn on/ off print function (default: True to turn on)
        :return:
        """
        # split data and calculate R_Squared & adj_R_Squared with full feature_columns
        X, y = StartMod.split_data(data, dependent_label, type_pd=False, split=False)

        try:
            regressor_ols = sm.OLS(endog=y, exog=X).fit()
        except TypeError:
            print("Data might have NaN_value, please clean them before building model")
            return data

        # backup data for not removing case
        orig_data = data
        if pr:
            print(data.columns)
            print("\nRSquared: ", regressor_ols.rsquared)
            print("\nAdj_RSquared: ", regressor_ols.rsquared_adj)
            print("\n", regressor_ols.summary())

        # calculate r_squared, adj_r_squared on the removing columns
        for rmc in rm_columns:
            print("\nRemove column: ", rmc)
            data = data.drop(rmc, axis=1)
            print(data.columns)
            X, y = StartMod.split_data(data, dependent_label, type_pd=False, split=False)

            regressor_ols = sm.OLS(endog=y, exog=X).fit()
            if print:
                print("R_Squared: ", regressor_ols.rsquared)
                print("Adj_R_Squared: ", regressor_ols.rsquared_adj)
                print("\n", regressor_ols.summary())

        # if yes, return data without column
        if rm:
            return data
        else:
            # return original data without any damages
            return orig_data

    @classmethod
    def feature_select_chi2(cls, data, k_features=4):
        array = data.values
        X = array[:, 0:len(array)]
        Y = array[:, len(array)]

        # feature extraction
        chosen = SelectKBest(score_func=chi2, k=k_features)
        fit = chosen.fit(X, Y)

        # print summarized scores
        np.set_printoptions(precision=3)
        print(fit.scores_)
        features = fit.transform(X)
        print(features[0:k_features+1, :])
        return features

    @classmethod
    def feature_test_chi2(cls, data):
        """
        determine whether input_variable and categorical output_variable are independent.
        data should exist 2 variables to create contigency table in format list of numpy_array
        (list of list) for column (feature_values as input_variable) and row (record_values as output_variable)
        """
        contigency_table = data.values
        print(contigency_table)
        stat, p, dof, expected = chi2_contingency(contigency_table)

        # Degree of freedom
        print('dof=%d' % dof)
        print(expected)

        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
        
        # interpret p-value
        alpha = 1.0 - prob
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')

    @classmethod
    def feature_hashing(cls, data):
        """
        Description: benefit on low-memory and speed up the performance

        References:
            http://scikit-learn.org/stable/modules/feature_extraction.html
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        # tbd
        pass

    @classmethod
    def feature_extraction(cls, data, dependent_label, tech='PCA', k_components=3):
        """
        Description: Dimensionality Reduction
            - Principal Component Analysis (PCA) to extract the most important independent variables (n_features)
                and reduce data from (n_features)-dimensions to k-dimensions (choose k-components)
            - Linear Discriminant Analysis (LDA)
        
        References:
            http://setosa.io/ev/principal-component-analysis/
            https://sebastianraschka.com/Articles/2014_python_lda.html
            http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param tech: choose the technique to redude dimension (default is PCA)
        :param k_components: number of principal components (for PCA, default is 3)
        :return: model, x_train, x_test, y_train, y_true
        """
        # TODO: choose the most best variance to get max possible total percentages
        x_train, x_test, y_train, y_true = StartMod.split_data(data, dependent_label)
        
        if tech: 
            pca = PCA(n_components = k_components)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
            print("Explained Variance: %s" % pca.explained_variance_ratio_)
            return pca, x_train, x_test, y_train, y_true
        else:
            lda = LinearDiscriminantAnalysis(n_components = k_components)
            x_train = lda.fit_transform(x_train, y_train)
            x_test = lda.transform(x_test)
            print("Explained Variance: %s" % lda.explained_variance_ratio_)
            return lda, x_train, x_test, y_train, y_true

    @classmethod
    def feature_engineering(cls, data, old_feature, new_feature, new_attributes, rm=False):
        """
        Description: renew data with new_feature using the new attributes_new_feature

        References:
            https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/

        :param data: pandas.core.frame.DataFrame
        :param old_feature:
        :param new_feature:
        :param new_attributes:
        :return: data (with new feature)
        """

        def find_attributes_feature(search_object, attributes):
            for attr in attributes:
                if attr in search_object:
                    # print(attr, search_object)
                    return attr
            return np.nan

        data[new_feature] = data[old_feature].map(lambda x: find_attributes_feature(x, new_attributes))
        if rm:
            data = data.drop(old_feature, axis=1)
        return data

    @classmethod
    def feature_engineering_merge_cols(cls, data, features, new_feature, datetime=False):
        """
        Description: merge many features into one new_feature (column)

        :param data: pandas.core.frame.DataFrame
        :param features: list of the merging features
        :param new_feature: name of merged new_feature (plus_combined)
        :param datetime: default is False (if True, then it will merge list of ['date', 'time'] features into
        new_feature (e.g. 'Date_Time')
        :return: data (with new column feature)
        """
        if datetime:
            # merge columns features ['Date','Time'] into new column new_feature ['Date_Time']
            data[new_feature] = data[features].apply(lambda x: pd.datetime.combine(*list(x)), axis=1)
        else:
            # initialize new feature column
            data[new_feature] = 0

            for feature in features:
                data[new_feature] = data[new_feature]+data[feature]

        return data.drop(features, axis=1)
    
    @classmethod
    def features_dim_reduction(cls):
        """
        Description: 
            use dimensionality reduction to reduce the dimension of data (features of data) into 2 or 3 dimension/ features
        """
        pass

    @classmethod
    def regularization(cls, data, dependent_label):
        """
        Description:
            apply technique of Ridge regression L2
            and Lasso (Least Absolute Shrinkage and Selection Operator) regression L1
            to avoid overfitting (when p>n, many more features than observations "wide matrix"):
            by (Lasso regression) shrinking some of the parameters to zero (therefore getting rid of some features) and
            by (Ridge regression) making some of the parameters very small without setting them to zero.
            by (Elastic net regression) which is basically a hybrid of ridge and lasso regression for big dataset

        1. Ridge:
            It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
            It reduces the model complexity by coefficient shrinkage.
            It uses L2 regularization technique.

        2. LASSO (Least Absolute Shrinkage Selector Operator):
            It selects the only some feature while reduces the coefficients of others to zero
            It is generally used when we have more number of features, because it automatically does feature selection.
            It uses L1 regularization technique

        3. Elastic Net Regression

        References:
            https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
            https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
            https://codingstartups.com/practical-machine-learning-ridge-regression-vs-lasso/
            https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :return:
        """
        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_true = StartMod.split_data(data, dependent_label, type_pd=False)

        # Ridge
        ridgeReg = Ridge(alpha=0.05, normalize=True)
        ridgeReg.fit(x_train, y_train)
        y_pred_rr = ridgeReg.predict(x_test)
        print("\nRidge coef_attribute", ridgeReg.coef_)
        print("Ridge independent term", ridgeReg.intercept_)
        print("Ridge evaluation using r-square: ", ridgeReg.score(x_test, y_true))

        # Lasso
        lassoReg = Lasso(alpha=0.3, normalize=True)
        lassoReg.fit(x_train, y_train)
        y_pred_lr = lassoReg.predict(x_test)
        print("\nLasso coef_attribute", lassoReg.coef_)
        print("Lasso independent term", lassoReg.intercept_)
        print("Lasso evaluation using r-square: ", lassoReg.score(x_test, y_true))
        
        # Elastic net
        enReg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
        enReg.fit(x_train, y_train)
        y_pred_er = enReg.predict(x_test)
        print("\nElastic net coef_attribute", enReg.coef_)
        print("Elastic net independent term", enReg.intercept_)
        print("Elastic net evaluation using r-square: ", enReg.score(x_test, y_true))

        print("\nMetrics report")
        print("Ridge regression: ", StartMod.metrics_report(y_true, y_pred_rr))
        print("Lasso regression: ", StartMod.metrics_report(y_true, y_pred_lr))
        print("Elastic net regression: ", StartMod.metrics_report(y_true, y_pred_er))
        print("\nValidation")
        print("Ridge: ", StartMod.validation(ridgeReg, x_train, y_train))
        print("\nLasso: ", StartMod.validation(lassoReg, x_train, y_train))
        print("\nElastic net: ", StartMod.validation(enReg, x_train, y_train))

    @classmethod
    def lossClassification(cls, yHat, y):
        """
        Description: Classification Loss
            1. Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value 
            between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. 
            A perfect model would have a log loss of 0.
            
        References: 
            https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html´
            https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

        :param yHat: the predicted value (must be in range 0 and 1)
        :param y: the actual value (must be in range 0 and 1)
        """
        if yHat > 1 or y > 1 or yHat < 0 or y < 0:
            return np.nan

        # ----- 1. crossEntropy - Log Loss  -----
        if y == 1:
            return -np.log(yHat)
        else:
            return -np.log(1 - yHat)

    @classmethod
    def lossRegression(cls, y_pred, y_true, delta):
        """
        Description: Regression Loss
            L1-Loss MAE, L2-Loss MSE, Huber Loss (Smooth Mean), Log cosh Loss, Quantile Loss
            
            L1 loss MAE is more robust to outliers, but its derivatives are not continuous, making it inefficient to find the solution
            L2 loss MSE is sensitive to outliers, but gives a more stable and closed form solution
            
            Huber Loss (smooth Mean for MAE and MSE) is less sensitive to outliers in data than the squared error loss, 
            better choice than L1 loss and L2 loss. Hyperparameter, 𝛿 (delta), which can be tuned. 
            Huber loss approaches MAE when 𝛿 -> 0 and MSE when 𝛿 -> infinite

            Log-cosh Loss is the logarithm of the hyperbolic cosine of the prediction error, smoother than L2.

            Quantile loss functions turn out to be useful when predicting an interval instead of only point predictions.

            Root Mean Squared Error (RMSE): the square root of MSE
            Root Mean Squared Logarithmic Error (RMSLE): the square root of MSLE
        
        Reference:
            https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
            https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
        """
        # Regression Model Evaluation            
        print("L1-Loss Mean Absolute Error (MAE): \n", mean_absolute_error(y_true, y_pred))
        print("L2-Loss Mean Squared Error (MSE): \n", mean_squared_error(y_true, y_pred))
        print("Root Mean Squared Error (RMSE): \n", sqrt(mean_squared_error(y_true, y_pred)))

        print("Mean Squared Logarithmic Error (MSLE): \n", mean_squared_log_error(y_true, y_pred))
        print("Root Mean Squared Logarithmic Error (RMSLE): \n", sqrt(mean_squared_log_error(y_true, y_pred)))
        
        huber_loss = np.sum(np.where(np.abs(y_true-y_pred) < delta , 1/2*((y_true-y_pred)**2), delta*np.abs(y_true - y_pred) - 1/2*(delta**2)))
        log_cosh_loss = np.sum(np.log(np.cosh(y_pred - y_true)))

        print("Huber Loss: \n ", huber_loss)
        print("Cosh Loss: \n ", log_cosh_loss)
        
        print("R2 Score (coefficient of determination): \n", r2_score(y_true, y_pred))
        print("Adjusted R2 Score: tbd. ")   
        
    @classmethod
    def validate_classification(cls, model, x_val, y_val, classification_metrics='accuracy', tune=False, vis=True):
        """
        Description: apply K-Fold Cross_Validation to estimate the model (classification) Bias vs Variance

        Debugging a learning algorithm:
            - Get more training data
            - Try smaller sets of features
            - Try getting additional features
            - Try adding polynomial features
            - Try decreasing regularization parameter
            - Try increasing regularization parameter

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
            http://scikit-learn.org/stable/modules/cross_validation.html
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
            https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
            https://www.jeremyjordan.me/hyperparameter-tuning/
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            http://scikit-learn.org/stable/modules/grid_search.html
            
        :param model: trained model
        :param x_val: x_validation_feature_values
        :param y_val: y_validation_categorical_values
        :param classification_metrics: classification accuracy {Accuracy, Logistic Loss, Area under ROC Curve}
        :param parameters (used for tuning hyperparameters) default is []
        :param tune (turn on grid search method to find the best parameters for model) default is False
        :return:
        """        
        # Evaluate using Cross validation with k-parts with default n_splits = 10, random_state = 7
        kfold = KFold(n_splits=10, random_state=7)

        if classification_metrics is 'accuracy':
            # Cross Validation Classification Accuracy as the number of correct predictions
            scores = cross_val_score(estimator=model, X=x_val, y=y_val, cv=kfold, scoring='accuracy')
        elif classification_metrics is 'neg_log_loss':
            # Cross Validation Classification Logistic Loss (logloss) for evaluating the predictions of probabilities
            scores = cross_val_score(estimator=model, X=x_val, y=y_val, cv=kfold, scoring='neg_log_loss')
        else:
            # Cross Validation Classification ROC AUC for binary classification problem
            scores = cross_val_score(estimator=model, X=x_val, y=y_val, cv=kfold, scoring='roc_auc')

        print("\nAccuracy")
        print("\nCross_validated scores: ", scores)        
        print("\nMean of cross_validated scores: %.3f%%" % (scores.mean()*100.0))
        print("\nStandard Deviation of cross_validated scores : %.3f%%" % (scores.std()*100.0))               

        # Plot cross_validated predictions
        predictions = cross_val_predict(model, x_val, y_val, cv=6)

        if vis:
            # print(len(x_val), len(y_val), len(predictions))
            plt.scatter(x_val, y_val, 'r--', x_val, predictions, 'bs')

        # calculate R_Squared to measure the cross_predicted accuracy of the model
        accuracy = r2_score(y_val, predictions)
        print("\nCross_predicted accuracy:", accuracy)
    
        if tune:
            alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
            param_grid = dict(alpha = alphas)

            # setup grid search input parameters to tune the hyper parameter e.g. n_jobs=-1 for large data set
            gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
            gs.fit(x_val, y_val)           
            
            print("Grid Search Parameter - Best accuracy: ", gs.best_score_)
            print("Grid Search Parameter - Best parameters: ", gs.best_params_)
            print("Grid Search Parameter - Best estimator alpha: ", gs.best_estimator_.alpha)
            print("Grid Search Parameter - Best: %f using %s" % (gs.best_score_, gs.best_params_))

            means = gs.cv_results_['mean_test_score']
            stds = gs.cv_results_['std_test_score']
            params = gs.cv_results_['params']

            for mean, stdev, param in zip(means, stds, params):
                print("Mean: %f, Standard Deviation: (%f), Parameters: %r" % (mean, stdev, param))

            # Total 100 iterations are performed randomly with alpha values between 0 and 1
            param_grid = {'alpha': uniform()}            
            rs = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, random_state=7)
            rs.fit(x_val, y_val)
            print("Random Search Parameter - Best Score: ", rs.best_score_)
            print("Random Search Parameter - Best estimator alpha: ", rs.best_estimator_.alpha) 

    @classmethod
    def report_classification_metrics(cls, y_true, y_pred, cat_lab):
        """
        Description: measure the quality of the models (comparing results before and after running prediction)

            Binary Classification:
                Accuracy = (TP + TN) / (TP + TN + FP + FN)
                Precision = TP / (TP + FP)
                Recall = TP / (TP + FN)
                F1 Score = (2 * Precision * Recall) / (Precision + Recall)
            
            Regression models: 
                Mean absolute error, Mean Squared error and R2 score

        References:
            https://medium.com/acing-ai/how-to-evaluate-regression-models-d183b4f5853d
            http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
            http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

        :param y_true: the truth values
        :param y_pred: the predicted values
        :param cat_lab: categorical label name used for classification 
        :return:
        """
        
        # Classification Model Evaluation
        print("Classification Report: \n", classification_report(y_true, y_pred, cat_lab=cat_lab))
        print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred, labels=np.unique(y_true)))        
        print("\nAccuracy Score: \n", accuracy_score(y_true, y_pred))

        if len(np.unique(y_true))==2:
            print("binary")
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
        else:
            print("set average")
            prec = precision_score(y_true, y_pred, average='micro')
            rec = recall_score(y_true, y_pred, average='micro')

        print("\nPrecision Score: \n", prec)
        print("\nRecall Score: \n", rec)
        print("\nF-Score: \n", 2*prec*rec/ (prec+rec))

    @classmethod
    def report_regression_metrics(cls, model, x_val, y_val, regression_metrics):
        """ Description: Metrics for evaluating predictions on regression machine learning problems """
        
        # Evaluate using Cross validation with k-parts with default n_splits = 10
        kfold = KFold(n_splits=10, random_state=7)

        if regression_metrics is 'neg_mean_absolute_error':
            # Mean Absolute Error MAE 
            scores = cross_val_score(estimator=model, X=x_val, y=y_val, cv=kfold, scoring='neg_mean_absolute_error')
        elif regression_metrics is 'neg_mean_squared_error':
            # Mean Squared Error MSE
            scores = cross_val_score(estimator=model, X=x_val, y=y_val, cv=kfold, scoring='neg_mean_squared_error')
        else:
            # R2 Squared
            scores = cross_val_score(estimator=model, X=x_val, y=y_val, cv=kfold, scoring='r2')

        print("\nAccuracy")
        print("\nCross_validated scores: ", scores)        
        print("\nMean of cross_validated scores: %.3f%%" % (scores.mean()*100.0))
        print("\nStandard Deviation of cross_validated scores : %.3f%%" % (scores.std()*100.0))   

        print( 'skewness of normal distribution (should be 0): {}'.format(skew(y_val)))        
        print('Kurtosis for normal distribution (normal 0.0)', kurtosis(y_val, fisher = True))
        print('Kurtosis for normal distribution Pearson’s definition is used (normal 3.0)', kurtosis(y_val, fisher = False))

    # @classmethod
    # def bootstrap_eval(cls, model, data, number_repeats):
    #     # prepare bootstrap sample
    #     boot_train = resample(data, replace=True, n_samples=4, random_state=1)
    #     print('Bootstrap Sample: %s' % boot_train)
    #     # out of bag observations
    #     oob_test = [x for x in data if x not in boot_train]
    #     print('OOB Sample: %s' % oob_test)

        # fit model to data
        # model.fit(boot_train)        
        # statistics = [evaluate(model, oob_test) for i in range(number_repeats)]
    @classmethod
    def compLowUpCI(cls, number_samples, number_correct, significant_level):
        """
        Description: 
            lower and upper bounds on the model classification'S accuracy
        :param number_samples: size of data
        :param number_correct: the number of corrected prediction on total samples/ data
        :param significant_level: 90%, 95%, 98%, 99%
        :return lower, upper
        """
        lower, upper = proportion_confint(number_correct, number_samples, 1-significant_level)
        print('lower=%.3f, upper=%.3f' % (lower, upper))
        return lower, upper