#####################################################################################
#                                                                                   #
#                PLOT A LIVE GRAPH IN A PYQT WINDOW                                 #
#                EXAMPLE 1 (modified for extra speed)                               #
#               --------------------------------------                              #
# This code is inspired on:                                                         #
# https://matplotlib.org/3.1.1/gallery/user_interfaces/embedding_in_qt_sgskip.html  #
# and on:                                                                           #
# https://bastibe.de/2013-05-30-speeding-up-matplotlib.html                         #
#                                                                                   #
#####################################################################################

from __future__ import annotations
from typing import *
import sys
import os
from matplotlib.backends.qt_compat import QtCore, QtWidgets
# from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
import matplotlib.figure as mpl_fig
import numpy as np

class ApplicationWindow(QtWidgets.QMainWindow):
    '''
    The PyQt5 main window.

    '''
    def __init__(self):
        super().__init__()
        # 1. Window settings
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("Matplotlib live plot in PyQt - example 1 (modified for extra speed)")
        self.frm = QtWidgets.QFrame(self)
        self.frm.setStyleSheet("QWidget { background-color: #eeeeec; }")
        self.lyt = QtWidgets.QVBoxLayout()
        self.frm.setLayout(self.lyt)
        self.setCentralWidget(self.frm)

        # 2. Place the matplotlib figure
        # self.myFig = MyFigureCanvas(x_len=200, y_range=[0, 100], interval=1)
        self.myFig = ScatterCanvas(x_range=[-10, 10], y_range=[-10, 10], interval=1)
        self.lyt.addWidget(self.myFig)

        # 3. Show
        self.show()
        return

class MyFigureCanvas(FigureCanvas):
    '''
    This is the FigureCanvas in which the live plot is drawn.

    '''
    def __init__(self, x_len:int, y_range:List, interval:int) -> None:
        '''
        :param x_len:       The nr of data points shown in one plot.
        :param y_range:     Range on y-axis.
        :param interval:    Get a new datapoint every .. milliseconds.

        '''
        super().__init__(mpl_fig.Figure())
        # Range settings
        self._x_len_ = x_len
        self._y_range_ = y_range

        # Store two lists _x_ and _y_
        self._x_ = list(range(0, x_len))
        self._y_ = [0] * x_len

        # Store a figure ax
        self._ax_ = self.figure.subplots()
        self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1]) # added
        self._line_, = self._ax_.plot(self._x_, self._y_)                  # added
        self.draw()                                                        # added

        # Initiate the timer
        self._timer_ = self.new_timer(interval, [(self._update_canvas_, (), {})])
        self._timer_.start()
        return

    def _update_canvas_(self) -> None:
        '''
        This function gets called regularly by the timer.

        '''
        self._y_.append(round(get_next_datapoint(), 2))     # Add new datapoint
        self._y_ = self._y_[-self._x_len_:]                 # Truncate list y

        # Previous code
        # --------------
        # self._ax_.clear()                                   # Clear ax
        # self._ax_.plot(self._x_, self._y_)                  # Plot y(x)
        # self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1])
        # self.draw()

        # New code
        # ---------
        self._line_.set_ydata(self._y_)
        self._ax_.draw_artist(self._ax_.patch)
        self._ax_.draw_artist(self._line_)
        self.update()
        self.flush_events()
        return


class ScatterCanvas(FigureCanvas):
    def __init__(self, x_range:list, y_range:List, interval:int) -> None:
        '''
        :param x_range:       
        :param y_range:     Range on y-axis.
        :param interval:    Get a new datapoint every .. milliseconds.

        '''
        super().__init__(mpl_fig.Figure())
        # Range settings
        self._x_range_ = x_range
        self._y_range_ = y_range

        # Store two lists _x_ and _y_
        self._x_ = [0]
        self._y_ = [0]

        # Store a figure ax
        self._ax_ = self.figure.subplots()
        self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1]) # added
        self._ax_.set_xlim(xmin=self._x_range_[0], xmax=self._x_range_[1]) # added
        self._scatter_= self._ax_.scatter(self._x_, self._y_)             # added
        self.draw()                                                        # added

        # Initiate the timer
        self._timer_ = self.new_timer(interval, [(self._update_canvas_, (), {})])
        self._timer_.start()
        return

    def _update_canvas_(self):
        x, y = np.squeeze(np.random.randn(1, 2)).tolist()
        self._x_.append(x)
        self._y_.append(y)
        # self._y_.append(round(get_next_datapoint(), 2))     # Add new datapoint
        # self._y_ = self._y_[-self._x_len_:]                 # Truncate list y

        # Previous code
        # --------------
        # self._ax_.clear()                                   # Clear ax
        # self._ax_.plot(self._x_, self._y_)                  # Plot y(x)
        # self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1])
        # self.draw()

        # New code
        # ---------
        # self._scatter_.set_ydata(self._y_)
        # self._scatter_.set_xdata(self._x_)
        # self._ax_.draw_artist(self._ax_.patch)
        # self._ax_.draw_artist(self._scatter_)
        # self.update()
        # self.flush_events()

        # my code
        self._ax_.clear()
        self._ax_.scatter(self._x_, self._y_)
        self.draw()

        return


# Data source
# ------------

def get_next_point():

    return np.squeeze(np.random.randn(1, 2)).tolist()

n = np.linspace(0, 499, 500)
d = 50 + 25 * (np.sin(n / 8.3)) + 10 * (np.sin(n / 7.5)) - 5 * (np.sin(n / 1.5))
i = 0
def get_next_datapoint():
    global i
    i += 1
    if i > 499:
        i = 0
    return d[i]

if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    qapp.exec_()
