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


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.pipeline import make_pipeline
from startml import *


class StartMod(StartML):
    """
      Description: StartMod - Start Models
      Apply different Models in Machine Learning Regression and Classification
        k-NN, Decision Tree, SVM, Neural Network, etc.

      Start:
      jupyter notebook
      -> from startmod import *
      -> info_help
    """

    @classmethod
    def feature_hashing(cls, data):
        """
        Source: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
        :param data:
        :return:
        """
        pass

    @classmethod
    def feature_engineering(cls, data, old_feature, new_feature, attributes_new_feature):
        """
        Source: https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
        :param data:
        :param old_feature:
        :param new_feature:
        :param attributes_new_feature:
        :return:
        """

        def find_attributes_feature(search_object, attributes):
            for attr in attributes:
                if attr in search_object:
                    # print(attr, search_object)
                    return attr
            return np.nan

        data[new_feature] = data[old_feature].map(lambda x: find_attributes_feature(x, attributes_new_feature))
        return data

    @classmethod
    def one_hot_encode(cls, data):
        """
        Encode object-columns
        Source: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        This encoding is needed for feeding categorical data to many scikit-learn estimators,
        notably linear models and SVMs with the standard kernels.
        :param data:
        :return:
        """
        # labelencode to turn object-column into number-column
        # labelencoder_X = LabelEncoder()
        # X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

        # specify index of the column which we want to encode (this case: column 0)
        # onehotencoder to turn number-column into sparse matrix
        # onehotencoder = OneHotEncoder(categorical_features=[0])
        # X = onehotencoder.fit_transform(X).toarray()

        return pd.get_dummies(data)


    @classmethod
    def score_dataset(cls):
        """
        measure the quality of the models (comparing results before and after running prediction)
        Source: https://www.kaggle.com/dansbecker/handling-missing-values
        :return:
        """
        print(StartML.kwargs)


StartMod.score_dataset()