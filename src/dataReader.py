# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   DataLoader.py
@Time    :   2021/12/03 16:24:59
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   从文件中读取数据
-------------------------
'''

import os
from typing import List, Any


class DataReaderForTest:
    def __init__(self, filePath):
        self.filePath = filePath
        self.data = []
        self._cvrt2list()

    def _cvrt2list(self):
        with open(self.filePath, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                amplitude = float(line.split()[0])
                self.data.append(amplitude)

    def __call__(self, *args: Any, **kwds: Any) -> List:
        return self.data

    
if __name__ == "__main__":
    dataFile = "./data/zn.txt"
    loader = DataReaderForTest(dataFile)
    a = loader()
    import numpy as np
    data = np.loadtxt(dataFile)
    print(data[:100])