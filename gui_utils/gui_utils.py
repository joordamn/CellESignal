# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   gui_utils.py
@Time    :   2022/02/23 20:39:39
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   gui window class and util functions
-------------------------
'''


from queue import Queue
import sys
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.figure as mpl_fig
from matplotlib.patches import Rectangle
import numpy as np

from utils.utils import save_and_plot

class ApplicationWindow(QtWidgets.QMainWindow):
    '''
    The PyQt5 main window.
    '''
    def __init__(self, save_folder, cfg):
        super().__init__()
        # 1. Window settings
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("Scatter Plot of cell signal")
        self.frm = QtWidgets.QFrame(self)
        self.frm.setStyleSheet("QWidget { background-color: #eeeeec; }")
        self.lyt = QtWidgets.QVBoxLayout()
        self.frm.setLayout(self.lyt)
        self.setCentralWidget(self.frm)
        self.save_folder = save_folder
        self.cfg = cfg

        # 2. Place the matplotlib figure
        self.canvas = ScatterCanvas(
            x_range=[0, 100], 
            y_range=[0, 0.008], 
            save_folder=self.save_folder,
            cfg=self.cfg,
            )
        self.lyt.addWidget(self.canvas)

        # 3. Show

        self.show()
        return


class ScatterCanvas(FigureCanvas):
    def __init__(self, x_range:list, y_range:list, save_folder, cfg) -> None:
        '''
        :param x_range:     Range on x-axis.
        :param y_range:     Range on y-axis.
        :param interval:    Get a new datapoint every .. milliseconds.

        '''
        super().__init__(mpl_fig.Figure())
        self.save_folder = save_folder
        # gate范围
        self.y_live_pos = cfg.sorting_gate['ppVal'][0]
        self.x_live_pos = cfg.sorting_gate['travel_time'][0]
        self.gate_live_pos = [self.x_live_pos[0], self.y_live_pos[0]]  # 活细胞gate rectangle 起点
        self.gate_live_w = self.x_live_pos[1] - self.x_live_pos[0]  # 活细胞 gate rectangle w
        self.gate_live_h = self.y_live_pos[1] - self.y_live_pos[0]  # 活细胞 gate rectangle h

        self.y_dead_pos = cfg.sorting_gate['ppVal'][1]
        self.x_dead_pos = cfg.sorting_gate['travel_time'][1]
        self.gate_dead_pos = [self.x_dead_pos[0], self.y_dead_pos[0]]  # 活细胞gate rectangle 起点
        self.gate_dead_w = self.x_dead_pos[1] - self.x_dead_pos[0]  # 活细胞 gate rectangle w
        self.gate_dead_h = self.y_dead_pos[1] - self.y_dead_pos[0]  # 活细胞 gate rectangle h

        self.rectangles = {
            'live cell gate': Rectangle(self.gate_live_pos, self.gate_live_w, self.gate_live_h, fc='None', ec='g'),
            'dead cell gate': Rectangle(self.gate_dead_pos, self.gate_dead_w, self.gate_dead_h, fc='None', ec='r'),
        }
        # Range settings
        self._x_range_ = x_range
        self._y_range_ = y_range

        # Store two lists _x_ and _y_
        self._x_ = [0]
        self._y_ = [0]

        # Store a figure ax
        self._ax_ = self.figure.subplots()
        self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1])
        self._ax_.set_xlim(xmin=self._x_range_[0], xmax=self._x_range_[1])
        self._ax_.set_xlabel("Traveling Time (ms)")
        self._ax_.set_ylabel("Measured Voltage (V)")
        self._scatter_= self._ax_.scatter(self._x_, self._y_)

        # draw annotation and gate rect
        for gate_rect in self.rectangles:
            self._ax_.add_artist(self.rectangles[gate_rect])
            text_x, text_y = self.rectangles[gate_rect].get_xy()
            self._ax_.annotate(gate_rect, (text_x, text_y))

        self.draw()   

        # Initiate the timer
        # self._timer_ = self.new_timer(interval, [(self._update_canvas_, (), {})])
        # self._timer_.start()
        return

    def _update_canvas_(self, draw_package):

        signal = draw_package["raw_signal"]
        timestamp = draw_package["timestamp"]
        borders = draw_package["borders"]
        pred_prob = draw_package["pred_prob"]
        flag = draw_package["flag"]
        count = draw_package["count"]

        save_and_plot(
            save_folder=self.save_folder,
            raw_signal=signal,
            borders=borders,
            pred_prob=pred_prob,
            timestamp=timestamp,
            count=count,
            flag=flag,
        )

        x = draw_package["traveltime"]
        y = draw_package["ppVal"]
        self._x_.append(x)
        self._y_.append(y)

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
        self._ax_.scatter(self._x_, self._y_, s=2)
        self._ax_.set_xlabel("Traveling Time (ms)")
        self._ax_.set_ylabel("Measured Voltage (V)")
        for gate_rect in self.rectangles:
            self._ax_.add_artist(self.rectangles[gate_rect])
            text_x, text_y = self.rectangles[gate_rect].get_xy()
            self._ax_.annotate(gate_rect, (text_x, text_y))

        self.draw()

        return
        

if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    qapp.exec_()


    