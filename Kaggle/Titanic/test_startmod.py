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

import unittest
from startmod import *


class StartModTestCase(unittest.TestCase):
    """
    http://pyunit.sourceforge.net/pyunit.html
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def testFeature_engineering():
        new_columns = StartMod.feature_engineering(train_data, 'Name', 'Title',
                                                   attributes_new_feature=['Mrs', 'Mr', 'Miss', 'Ms']).columns

        assert np.array_equal(train_data.columns, new_columns), 'feature engineering not successful'

    @staticmethod
    def testFeature_engineering_merge_cols():
        new_columns = StartMod.feature_engineering_merge_cols(train_data, ['SibSp', 'Parch'], 'FamilySize').columns

        assert np.array_equal(new_columns, ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Ticket',
                                            'Fare', 'Cabin', 'Embarked', 'Title', 'FamilySize']), \
            'feature engineering merge columns not successful'

    # @staticmethod
    # def testEncode_label_column_invalid():
    #     x = StartMod.encode_label_column(train_data, 'FamilySize')
    #     assert np.array_equal(x, []), "encoding not successful"

    # @staticmethod
    # def testEncode_label_column_valid():
    #     X = StartMod.encode_label_column(train_data, 'Sex')
    #     assert ((X[:, 4]==1.0).sum(), (X[:, 4]==0.0).sum()) == (577, 314), "encoding not successful"


if __name__ == '__main__':
    unittest.main()
