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
from utils.utils import parsePeaks


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
    def __init__(
        self, 
        classifier_weight, 
        segmentator_weight, 
        device, 
        cfg,
        interpolate_length=256,
        ):
        self.detector = PeakDetector(classifier_weight, segmentator_weight, device, interpolate_length)
        self.exp_mode = cfg.exp_mode
        self.cfg = cfg

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
        
    def decision(self, signal_pack1: tuple, signal_pack2: tuple=None):
        """_summary_

        Args:
            signal_pack1 (tuple): ([signal1], [borders1])
            signal_pack2 (tuple): ([signal1], [borders1])
        Returns:
            decision (int): channel state
            flag (str): flag to be printed on screen
        """
        signal1, borders1 = signal_pack1
        if signal_pack2:
            signal2, borders2 = signal_pack2

        # 根据sorting gate进行决策
        decision, flag, traveltime, ppVal = self.signal_parse_with_decision(signal1, borders1)
        # decision = random.randint(1, 2)
        return decision, flag, traveltime, ppVal

    def signal_parse_with_decision(self, signal_slice, borders):
        """parse the signal and make decision

        Args:
            signal_slice (list): 信号切片list
            borders (list): 峰值范围list

        Returns:
            decision (int): 决策 0/1/2
            flag (str): 标记 live/dead/not in gate
        """
        ppVal_and_time = parsePeaks(signal_slice, borders)
        if not ppVal_and_time:
            return None, None, None, None
        travel_time, ppVal  = ppVal_and_time
        decision, flag = self.sorting_gate(travel_time, ppVal)
        return decision, flag, travel_time, ppVal

    def sorting_gate(self, travel_time, ppVal):
        # liveordead 模式
        if self.exp_mode == "liveordead":
            sorting_gate = self.cfg.sorting_gate
            live_pp_gate, dead_pp_gate = sorting_gate["ppVal"]
            live_time_gate, dead_time_gate = sorting_gate["travel_time"]
            # 活细胞gate范围
            if live_pp_gate[0] <= ppVal <= live_pp_gate[1]\
                and live_time_gate[0] <= travel_time <= live_time_gate[1]:
                return 1, "live cell"
            # 死细胞gate范围
            elif dead_pp_gate[0] <= ppVal <= dead_pp_gate[1]\
                and dead_time_gate[0] <= travel_time <= dead_time_gate[1]:
                return 2, "dead cell"
            # 未进入gate范围
            else:
                print(travel_time, ppVal)
                return 0, "not in sorting gate"
        # stiffness 模式
        elif self.exp_mode == "stiffness":
            pass
        else:
            raise RuntimeError("not implemented exp mode")
        


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
        

    