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

import unittest
from startml import *


class StartMLTestCase(unittest.TestCase):
    """
    http://pyunit.sourceforge.net/pyunit.html
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def testGet_value_column_index_Name():
        assert StartML.get_value_column_index(train_data, 'Name', 8) == \
               "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)", 'incorrect Name'

    @staticmethod
    def testGet_value_column_index_Fare():
        assert float(StartML.get_value_column_index(train_data, 'Fare', 10)) == 16.7, 'incorrect Fare price'

    @staticmethod
    def testNan_columns():
        assert StartML.nan_columns(train_data) == ['Age', 'Cabin', 'Embarked'], 'incorrect NaN columns'

    @staticmethod
    def testNan_rows():
        assert StartML.nan_rows(train_data).size == 8496, 'incorrect number of elements 708*12 in returning data object'

    @staticmethod
    def testPre_processing_columns():
        processed_train_data = StartML.process_nan_columns(train_data)

        if StartML.kwargs['drop_obj_col']:
            assert np.array_equal(processed_train_data.columns,
                                  ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']), \
                'the dropping columns are incorrect'

        elif StartML.kwargs['nan_drop_col']:
            assert np.array_equal(processed_train_data.columns,
                                  ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket',
                                   'Fare']), 'the dropping columns are incorrect'

        elif StartML.kwargs['nan_zero']:
            assert np.array_equal(processed_train_data.columns,
                                 ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                  'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
                                 ), 'the dropping columns are incorrect'
            assert StartML.get_value_column_index(processed_train_data, 'Age', 5) == 0, \
                'Incorrect replaced, value should be 0.0'

    @staticmethod
    def testPre_processing_rows():
        # processed_train_data = StartML.pre_processing_rows(train_data)
        # nan_cols = StartML.nan_columns(train_data)  # ['Age', 'Cabin', 'Embarked']

        if StartML.kwargs['nan_mean_neighbors']:
            assert StartML.mean_neighbors(train_data, 'Age', 32) == np.mean([40, 66]), 'Incorrect computed'


if __name__ == '__main__':
    unittest.main()
