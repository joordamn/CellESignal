# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   Processer.py
@Time    :   2021/12/03 15:49:12
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   用于接收数据，处理数据，判断波峰的类
-------------------------
'''


import numpy as np
import random
import time

from tools.infer import PeakDetector


class ProcesserForTest:
    def __init__(self, classifier_weight, segmentator_weight, device, interpolate_length=256):
        self.detector = PeakDetector(classifier_weight, segmentator_weight, device, interpolate_length)

    def __call__(self, input: np.ndarray):
        if type(input) is not np.ndarray:
            raise TypeError("input for recognizor is not ndarray")

        return self._process(input)

    def _process(self, input):
        if np.array_equal(input, np.empty(0)):
            decision = None
        else:
        # for test
            decision = random.randint(0,4)
        time.sleep(0.01) # 模拟处理时间 10毫秒
        return decision


class Processer():
    def __init__(self, classifier_weight, segmentator_weight, device, interpolate_length=256):
        self.detector = PeakDetector(classifier_weight, segmentator_weight, device, interpolate_length)
    
    def __call__(self, input: np.ndarray):
        """detect and return
        """
        if type(input) is not np.ndarray:
            raise TypeError("input for processer is not ndarray")

        return self._detect(input)

    def _detect(self, input):
        """
        infer the input with detector
        """

        label, border, pred_prob = self.detector(input)
        return label, border, pred_prob
        
    def decision(self, signal_pack1: tuple, signal_pack2: tuple):
        """_summary_

        Args:
            signal_pack1 (tuple): ([signal1], [borders1])
            signal_pack2 (tuple): ([signal1], [borders1])
        Returns:
            decision (int): channel state
        """
        signal1, borders1 = signal_pack1
        signal2, borders2 = signal_pack2

        # TODO change the method of decision making
        decision = random.randint(1, 2)
        return decision



if __name__ == "__main__":
    from dataReader import DataReaderForTest
    from recorder import RecorderForTest

    dataFile = "./data/zn.txt"
    loader = DataReaderForTest(dataFile)
    a = loader()

    recorder = RecorderForTest(9)

    recognizer = ProcesserForTest()

    for data in a[0:100]:
        data = np.asarray(data, dtype=np.float64)
        res = recorder(data)
        if res is not None:
            decision = recognizer(res)
            print(decision)
        

    