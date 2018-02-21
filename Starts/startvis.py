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
# import pyqtgraph
# import graphviz
import matplotlib.pyplot as plt

# import seaborn
# import plotly
# import plotnine
from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import rcParams
from Starts.startml import *


class StartVis(StartML):
    """
        Description: StartVis - Start Visualization
        Visualize data in different chart (Bar-Chart, Histograms, Scatter, TimeSeries, etc.)
        Reference: https://www.kaggle.com/learn/data-visualisation

        Start:
          jupyter notebook
          -> from startvis import *
          -> info_vis
    """

    @classmethod
    def vis_bar(cls, data, columns, x_label='', y_label='', title=''):
        """
        visualize the number of counted values in the given columns in bar-chart
        :param data: Pandas-DataFrame
        :param columns:
        :param x_label:
        :param y_label:
        :param title:
        :return:
        """
        for column in columns:
            # other options: line, area
            # data[column].head().value_counts().plot(kind='bar')
            if len(data[column].value_counts()) > 20:
                (data[column].value_counts() / len(data[column])).sort_index().plot.line()
            else:
                (data[column].value_counts()/ len(data[column])).sort_index().plot(kind='bar')
            plt.title("Bar Chart " + str(column) + " " + title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=0)
            plt.legend()
            plt.show()

    @classmethod
    def vis_hist(cls, data, columns, x_label='', y_label='', title='', func_filter=None):
        """
        Display Histogram of data and labels, with filter-function
        :param data: Pandas-DataFrame
        :param columns:
        :param func_filter: object type Pandas-Series
        :param x_label:
        :param y_label:
        :param title:
        :return: Histogram
        """

        try:
            for column in columns:
                if func_filter is None:
                    data[column].plot.hist()
                else:
                    data[func_filter][column].plot.hist()

                plt.title("Histogram " + str(column) + " " + title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.xticks(rotation=0)
                plt.legend()
                plt.show()
        except TypeError:
            print("No numeric data to plot")
            return

    @classmethod
    def vis_bar_groupby(cls, data, columns, group_by_column, x_label='', y_label='', title=''):
        """
        Visualize groupby-object in bar chart.
        :param data: Pandas-DataFrame
        :param columns:
        :param group_by_column:
        :param x_label:
        :param y_label:
        :param title:
        :return:
        """

        grouped_data = StartML.group_by_columns(data, columns, group_by_column)
        # grouped_data = data[columns].groupby(by=group_by_column)
        x = grouped_data.size()
        x.plot.bar(label=str(group_by_column))
        plt.title("Bar Chart group_by "+str(columns) + " " + title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=0)
        plt.legend()
        plt.show()

    @classmethod
    def vis_scatter(cls, data):
        pass

    @classmethod
    def vis_obj_predict(cls, x, y, obj_pred):
        """
        Visualizing result of the predicting object
        :param x:
        :param y:
        :param obj_pred:
        :return:
        """

        if x.shape == y.shape:
            plt.scatter(x, y, color='red')
        try:
            plt.plot(x, obj_pred.predict(x), color='blue')
            plt.title(type(obj_pred))
            plt.xlabel(str(type(x)))
            plt.ylabel(str(type(y)))
            plt.show()
        except AttributeError:
            print("Object has no Attribute predict, invalid object")

    @classmethod
    def vis_clustering(cls, data, y_clusters, x_label='', y_label='', ts=False):
        """
        plot clustering out with limited to data in 2 columns 0, 1)
        :param data:
        :param y_clusters:
        :param x_label:
        :param y_label:
        :param ts: set True if index is TimeSeries's type
        :return:
        """
        plt.figure(figsize=(16, 14))
        if ts:
            data.loc[:, 'Clusters'] = y_clusters
            for i in range(max(np.unique(y_clusters)) + 1):
                plt.scatter(StartML.lookup_value(data[['Clusters']], i, tup=False),
                            data.values[y_clusters == i, 0], s=10,
                            label='Cluster '+str(i))
            plt.xticks(rotation=45)
        else:
            for i in range(max(np.unique(y_clusters))+1):
                plt.scatter(data.values[y_clusters == i, 0], data.values[y_clusters == i, 1], s=10,
                            label='Cluster '+str(i))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        title = ''
        for col in data.columns:
            title = title + '-' + str(col)
        plt.title(title)
        plt.show()

    @classmethod
    def vis_basemap(cls, data, plot=False):
        """
        Source:
            https://matplotlib.org/basemap/
        :param data: Pandas-DataFrame (with geospatial coordinates 'Longitude' and 'Latitutde'
        :return:
        """
        try:
            latitude = data['Latitude'].tolist()
            longitude = data['Longitude'].tolist()
        except KeyError:
            print("Coordinates data Latitude and Longitude not sufficient")

        if 'Magnitude' in data.columns:
            mag = data['Magnitude'].tolist()
        else:
            mag = []

        earth = Basemap(projection='mill', resolution='c')
        x, y = earth(longitude, latitude)

        plt.figure(figsize=(16, 14))
        plt.title("Observation locations")

        if mag and not plot:
            print("Scatter function ...")
            earth.scatter(x, y, mag, marker='.', color='red')
        else:
            print("Plot function ...")
            earth.plot(x, y, "x", markersize=12, color='red')

        # setup basemap
        # earth.etopo(alpha=0.1)
        # earth.bluemarble(alpha=0.42)
        earth.drawcoastlines()
        earth.fillcontinents(color='coral', lake_color='aqua')
        earth.drawmapboundary()
        earth.drawcountries()

        plt.show()

    @classmethod
    def vis_contourf(cls, data):
        """
        Source:
            https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.contourf.html
        :param data:
        :return:
        """
        pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartVis": StartVis.__name__,
            "StartVis.vis_bar_groupby ": StartVis.vis_bar_groupby.__doc__,
            "StartVis.vis_scatter ": StartVis.vis_scatter.__doc__,
            "StartVis.vis_hist ": StartVis.vis_hist.__doc__,
            }

        return info


info_vis = StartVis.info_help()
