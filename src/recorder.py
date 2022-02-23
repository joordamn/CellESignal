# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   Recorder.py
@Time    :   2021/12/03 15:32:55
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   recorder类，用于收集、分包和发放data
-------------------------
'''

import numpy as np


class RecorderForTest:  # 针对每次返回单个数据点进行滑窗封装
    """
    用一个list容器一直接收data，
    每次调用该对象时都会返回数据，但不一定是有效数据，同时监测返回数据的次数
    当次数==数据包长度时，开始返回有效数据，同时次数重置为0
    并且对储存的数据进行滑窗，以便返回下一个带有overlap的数据包

    """
    def __init__(self, maxLen: int):
        # maxLen为数据包的长度
        # 数据包为一维的ndarray
        # overlap为两组数据包之间的重合度
        self.packLen = maxLen
        if self.packLen <= 3:
            raise ValueError("Check the length of the data pack setting, \
                it should be bigger than 4")
        self.streamData = []
        self._overlap = int(maxLen * 0.5)

        self.recordNum = 0

    def __call__(self, data: np.ndarray):
        """每次调用都会检查是否应该返回有效数据

        Args:
            data (np.ndarray): 输入的外部数据

        Raises:
            TypeError: 数据应为ndarray格式

        Returns:
            package (np.ndarray): 返回的一维数据包，长度为设置的maxLen
        """

        if type(data) is not np.ndarray:
            raise TypeError("stream data is not ndarray, please cheak the input stream")

        self._record(data)

        if len(self.streamData) < self.packLen:
            return np.empty(0)
        else:
            if self.recordNum >= self._overlap:
                self.recordNum = 0 # 当返回的数据包有值的时候就重置
                package = self._dataSliding()
                return package
            else:
                return np.empty(0)

    def _record(self, data):
        self.streamData.append(data)
        self.recordNum += 1
        pass

    def _dataSliding(self):
        """
        对存储的stream数据进行分包和滑窗
        取已存储的前几个数据进行打包返回
        删除overlap长度的前几个数据
        """
        if len(self.streamData) < self.packLen:
            raise Exception("Not enough data for sliding and package")

        package = self.streamData[0:self.packLen]
        package = np.asarray(package, dtype=np.float64)

        del self.streamData[:self._overlap]

        return package


class Recorder(RecorderForTest):  # 针对poll方法 每次返回一组数据进行封装
    def __init__(self, maxLen=100):
        super(Recorder, self).__init__(maxLen)
        self.streamData = {
            "timestamp": [],
            "signal": [],
        }
        
    def __call__(self, sample: dict):
        """
        sample(dict): should be requested by poll()
        sample{
            "timestamp": array([], dtype=uint64),
            "x": array([]),
            "y": array([]),
            "frequency": array([]),
            ...
        }
        """
        
        # sample data processing
        signals = np.abs(sample["x"] + 1j * sample["y"])
        timestamps = sample["timestamp"]
        
        # type transforming
        if len(signals.shape) > 1:
            signals = np.squeeze(signals).tolist()
        else:
            signals = signals.tolist()

        if len(timestamps.shape) > 1:
            timestamps = np.squeeze(timestamps).tolist()
        else:
            timestamps = timestamps.tolist()

        # write data into object container
        self._record(signals, timestamps)

        # return data according to the container length        
        if len(self.streamData["signal"]) < self.packLen + self._overlap:
            return None
        else:
            package = self._dataSliding()
            return package

    def _record(self, signal: list, timestamp: list):
        """
        将poll得到的数据存入streamData中
        TODO 用工厂模式精简代码
        """
        self.streamData["timestamp"].extend(timestamp)
        self.streamData["signal"].extend(signal)

    def _dataSliding(self):
        """
        把streamData分包成maxLen长度的多个数据包返回
        return:
            packages = [
                # package1
                {
                    "timestamp": [],
                    "signal": [],
                },
                # package2
                {
                    "timestamp": [],
                    "signal": [],
                },
                ...
            ]
        """
        if len(self.streamData["timestamp"]) < self.packLen:
            raise Exception("Not enough data for sliding and package")
        
        packages = []
        while len(self.streamData["timestamp"]) > self.packLen:
            package = {}
            for key in self.streamData.keys():
                package[key] = self.streamData[key][0:self.packLen]
                del self.streamData[key][:self._overlap]
            packages.append(package)

        return packages


if __name__ == "__main__":
    from src.dataReader import DataReaderForTest

    dataFile = "./data/zn.txt"
    loader = DataReaderForTest(dataFile)
    a = loader()

    recorder = RecorderForTest(50)
    
    counter = 0
    for data in a:
        data = np.asarray(data, dtype=np.float64)
        res = recorder(data)

        if not np.array_equal(res, np.empty(0)):
            counter += 1
    print(f"共打包了份{counter}有效数据")