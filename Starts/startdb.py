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

import json
import pymongo
from pandas import DataFrame


class StartDB(object):
    """
        Description: StartDB - Start Databases
        Operations-API to connect to Databases (Mongodb, PostgreSQL, Neo4j)

        References:
            https://www.mongodb.com/
            https://www.postgresql.org/
            https://neo4j.com/

        Start:
          jupyter notebook
          -> from startdb import *
          -> info_startdb
    """

    def __init__(self, db_type, db_path, db_port, db_name):
        self.db_type = db_type
        self.db_path = db_path
        self.db_port = db_port
        self.db_name = db_name

    def get_db_type(self):
        return self.db_type

    def get_db_path(self):
        return self.db_path

    def get_db_port(self):
        return self.db_port

    def get_db_name(self):
        return self.db_name

    def get_table_name(self):
        return self.table_name

    @staticmethod
    def info_help():
        info = {
            "info_help_StartDB": StartDB.__name__,
            }

        return info


class StartMongo(StartDB):
    """
    Reference:
        https://docs.mongodb.com/getting-started/python/
    """

    def __init__(self, db_path, db_port, db_name, coll_name):
        # Inherit attributes from StartDB
        StartDB.__init__(self, "MongoDB", db_path, db_port, db_name)
        self.coll_name = coll_name

        # initiate Mongo Client for MongoDB e.g.'localhost:27017'
        try:
            self.client = pymongo.MongoClient(self.db_path, self.db_port)
        except pymongo.errors.ConnectionFailure:
            print("Could not connect to Server")
            return

        # create database name
        self.db = self.client[self.db_name]

        # create collection 
        self.db_coll = self.db[self.coll_name]

    def get_coll_name(self):
        return self.coll_name

    def insert_data(self, data):
        """
        Insert data into database

        :param data:
        :return:
        """
        # convert data into json_format
        data_json = json.loads(data.to_json(orient='records'))

        # insert json-data into database
        self.db_coll.insert(data_json)

    def read_data(self, key_value=None):
        """
        Read data from MongoDB out

        Reference:
            https://mindey.com/blog/how_to_read_a_mongodb_into_pandas_dataframe

        :param key_value:
        :return:
        """
        try:
            if key_value:
                cursor = self.db_coll.find(key_value)
            else:
                cursor = self.db_coll.find()

            print("\nQuerying data from MongoDB .....\n")

            # convert and return data as dataframe
            df = DataFrame([observation for o, observation in enumerate(cursor)])
            return df

        except Exception as e:
            print(str(e))

    # Update data in MongoDB
    def update_data(self, key_value):
        self.db[self.coll_name].update(key_value)

    # Remove data in MongoDB
    def remove_data(self, coll_name, key_value):
        # self.db[coll_name].delete_many(key_value)
        self.db[coll_name].delete(key_value)

    # Clean data in MongoDB
    def clean_data(self, coll_name, coll=False):
        # delete the whole collection
        if coll:
            self.db[coll_name].drop()
        else:
            # delete all data in collection
            self.db[coll_name].remove()


info_startdb = StartDB.info_help()
