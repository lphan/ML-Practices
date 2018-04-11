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

from Starts.startmod import *
from Starts.startvis import *


class StartModREC(StartMod):
    """
        Description: StartModREC - Start Models Recommendation System
        regression, classification

        References:
            http://www.guidetodatamining.com/

            Start:
            jupyter notebook
            -> from startmodskl import *
            -> info_modskl
    """

    def __init__(self):
        pass

    @classmethod
    def find_nearest_neighbors_by_distance(cls, data, user_idx, k=5, manhattan=True):
        """
        return the most nearest neighbors based on computing Distance 'Manhattan Distance' or 'Euclidean Distance'
        to the given username

        :param data: pandas.core.frame.DataFrame (two feature columns representing x and y values)
        :param user_idx:
        :param k: number of nearest neighbors (default k=5)
        :param manhattan: (default True). Otherwise, apply Euclidean distance
        :return:
        """

        if len(data.columns) > 2:
            return data
        else:
            pop = StartML.pop_rows(data, user_idx, inplace=False)

            if manhattan and len(pop)==1:
                neighbors = [(i,
                              abs(pop[data.columns[0]].values - data.iloc[i][data.columns[0]])[0] +
                              abs(pop[data.columns[1]].values - data.iloc[i][data.columns[1]])[0])
                              for i in range(len(data))]
            else:
                # compute Euclidean distance
                neighbors = [(i,
                              np.sqrt(
                                  (abs(pop[data.columns[0]].values - data.iloc[i][data.columns[0]])[0]**2) +
                                  (abs(pop[data.columns[1]].values - data.iloc[i][data.columns[1]])[0]**2)
                              ))
                             for i in range(len(data))]

            # sort value and return the first 5 values
            nnb = pd.DataFrame(data=neighbors, columns=['idx', 'distance'])
            nnb.set_index('idx', inplace=True)
            nnb.sort_values(by='distance', ascending=True, inplace=True)
            return nnb[1:k+1]

    @classmethod
    def find_nearest_neighbors_by_correlation(cls, data, k=5):
        """
        return the most nearest neighbors based on computing 'Pearson' correlation

        :param data:
        :param k:
        :return:
        """
        pass

    @classmethod
    def pearson(cls, data):
        """
        compute Pearson_correlation on data with two columns representing x values and y values
        :param data:
        :return:
        """
        numerator = 0
        if len(data) == 0:
            return 0

        if len(data.columns) > 2:
            return data
        else:
            x = data[data.columns[0]].values
            y = data[data.columns[1]].values
            for i in range(len(data)):
                numerator = numerator + data[data.columns[0]][i] * data[data.columns[1]][i]
            numerator = numerator - sum(x) * sum(y) / len(data)

            data_squared = data.apply(lambda x: x ** 2)
            sum_squared_x = sum(data_squared[data_squared.columns[0]])
            sum_squared_y = sum(data_squared[data_squared.columns[1]])
            x_sum_squared = (sum(x)) ** 2 / len(data)
            y_sum_squared = (sum(y)) ** 2 / len(data)

            denominator = np.sqrt(sum_squared_x - x_sum_squared) * np.sqrt(sum_squared_y - y_sum_squared)
            print("numerator ", numerator)
            print("denominator ", denominator)
            if denominator == 0:
                return 0

            r = numerator / denominator

            return r

    @classmethod
    def find_nearest_neighbors_by_similarity(cls, data, user_idx, k=5, Cosine=True):
        """
        return the most nearest neighbors based on computing 'Cosine' Similarity
        """
        pass

    @classmethod
    def find_similar_items(cls):
        """
        Item-based collaborative filtering (Adjusted Cosine Similarity) recommendation algorithms

        Reference:
            http://www.grouplens.org/papers/pdf/www10_sarwar.pdf

        """
        pass

    @classmethod
    def recommend_from_neighbors(cls, data, user_idx, nnd=True):
        """
        return the most closest products for 'username' based on the similar information from neighbors
        if nnd choose function 'findNearestNeighborsDistance'
        otherwise choose function 'findNearestNeighborsCorrelation'
        """
        pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModREC": StartMod.__name__,
            "StartModREC.find_nearest_neighbors_by_distance": StartModREC.find_nearest_neighbors_by_distance.__doc__,
            "StartModREC.find_nearest_neighbors_by_correlation": StartModREC.find_nearest_neighbors_by_correlation.__doc__,
            "StartModREC.find_nearest_neighbors_by_similarity": StartModREC.find_nearest_neighbors_by_similarity.__doc__,
            "StartModREC.recommend_from_neighbors": StartModREC.recommend_from_neighbors.__doc__,
            }
        return info


info_modskl = StartMod.info_help()