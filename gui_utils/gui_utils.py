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


from PyQt5 import QtCore, QtWidgets


class AbstractWindow(QtWidgets.QMainWindow):
    pass

class MainWindow(AbstractWindow):
    def __init__(self, xml_path, dll_path, model_path):
    
        super().__init__()
        pass

    def _init_ui(self):
        # canvas layout (camera window and signal monitor window)
        
        pass

    