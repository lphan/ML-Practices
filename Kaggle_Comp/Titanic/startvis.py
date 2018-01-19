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
# import pyqtgraph
# import graphviz
import matplotlib.pyplot as plt
# import seaborn
# import plotly
# import plotnine
from matplotlib.pylab import rcParams
from startml import *
rcParams['figure.figsize'] = 15, 6


class StartVis(StartML):
    """
      Description: StartVis - Start Visualization
      Visualize data in different chart (Bar-Chart, Histograms, Scatter, TimeSeries, etc.)

      Start:
          jupyter notebook
          -> from startvis import *
          -> info_vis
    """

    @classmethod
    def vis_bar(cls, data, columns, x_label='', y_label='', title=''):
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
        :param data:
        :param columns:
        :param func_filter:
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
        grouped_data = data[columns].groupby(by=group_by_column)
        grouped_data.plot(kind='bar', label=str(group_by_column))
        plt.title("Bar Chart group_by "+str(columns) + " " + title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=0)
        plt.legend()
        plt.show()

    @classmethod
    def vis_scatter(cls, data):
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
