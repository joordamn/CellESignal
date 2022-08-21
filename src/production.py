# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   production.py
@Time    :   2022/01/05 21:05:07
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   生产者消费者线程函数
-------------------------
'''
import os, sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import keyboard
from queue import Queue
import numpy as np
import time
from threading import Thread

from src.recorder import Recorder
from src.processer import Processer
from utils.utils import save_and_plot, post_plot, save_and_plot_doubleChannel
from utils.daqSessionCreator import DaqSession


def dataProduction(recorderIns: Recorder, daq, q_in: Queue, path, t_end=500):
    """生产者线程函数

    """

    counter = 0
    counter2 = 0
    curr_time = time.time()
    daq.sync()
    daq.subscribe(path)
    time.sleep(0.05)
    # while time.time() - curr_time < t_end: # or len(recorderIns.streamData["timestamp"]) > recorderIns._overlap:
    while True:

        start_t = time.time()
        poll = daq.poll(0.02, 200, 0, True)
        sample = poll[path]
        read_t = time.time() - start_t
        # print("read time takes {:.5f}, get data of length {}".format(read_t, len(sample["timestamp"])))

        packages = recorderIns(sample)
        # print("recorder容器中剩余{}长度的数据".format(len(recorderIns.streamData["timestamp"])))
        if packages:
            for package in packages:
                q_in.put(package)
                counter += 1

        counter2 += 1
        
    # 生产结束标志位
        if keyboard.is_pressed("/"):
            q_in.put("finish")
            print(f"共打包了{counter}次数据, 循环了{counter2}次")

            daq.unsubscribe(path)


def dataProcess(processer: Processer, q_out: Queue, save_folder, ser=None, plot_online=False):
    # 消费者线程函数
    package_counter = 0
    signal_counter = 0
    while True:
        packageGet = q_out.get()
        if packageGet == "finish":
            break
        start_t = time.time()
        signal = packageGet["signal"]
        timestamp = packageGet["timestamp"]
        label, borders, pred_prob = processer(np.asarray(signal))
        processing_t = time.time() - start_t

        if label == 1:
            signal_counter += 1
            decision, flag = processer.decision((signal, borders))
            save_and_plot(
                save_folder=save_folder, 
                raw_signal=signal, 
                borders=borders, 
                pred_prob=pred_prob,
                timestamp=timestamp, 
                count=signal_counter, 
                flag=flag,
                plot_online=plot_online,
                )

            print(f"signal number:{signal_counter} || {flag}")
        # print("Processing takes {:.5f} seconds".format(processing_t))

            # 向串口写入1-2
            # 0 -> 类别0  未进入sorting gate范围
            # 1 -> 类别1  状态1
            # 2 -> 类别2  状态2
            if ser and decision:
                ser.write(str(decision).encode('utf-8'))      

        package_counter += 1
    print(f"共接收到{package_counter}次数据包, {signal_counter}次信号")
    if not plot_online:
        post_plot(json_folder=save_folder)


class DataProducer(Thread):
    def __init__(
        self, 
        dataRecorder1: Recorder, 
        dataRecorder2: Recorder, 
        daqSession1: DaqSession,
        daqSession2: DaqSession, 
        q_in1: Queue, 
        q_in2: Queue, 
        ):
        super(DataProducer, self).__init__()
        self.dataRecorder1 = dataRecorder1
        self.dataRecorder2 = dataRecorder2
        self.daq1 = daqSession1.daq
        self.daq2 = daqSession2.daq
        self.q_in1 = q_in1
        self.q_in2 = q_in2
        self.path1 = daqSession1.path
        self.path2 = daqSession2.path
        
    def run(self):
        counter_all, counter1, counter2 = 0, 0, 0
        curr_time = time.time()
        self.daq1.sync()
        self.daq1.subscribe(self.path1)
        self.daq2.sync()
        self.daq2.subscribe(self.path2)

        time.sleep(0.05)
        while True:

            # start_t = time.time()
            # daq1 获取数据
            poll1 = self.daq1.poll(0.02, 200, 0, True)
            sample1 = poll1[self.path1]
            # daq2 获取数据
            poll2 = self.daq2.poll(0.02, 200, 0, True)
            sample2 = poll2[self.path2]

            # read_t = time.time() - start_t
            # print("read time takes {:.5f}, get data of length {}".format(read_t, len(sample["timestamp"])))

            # print("recorder容器中剩余{}长度的数据".format(len(recorderIns.streamData["timestamp"])))
            packages1 = self.dataRecorder1(sample1)
            if packages1:
                for package in packages1:
                    self.q_in1.put(package)
                    counter1 += 1

            packages2 = self.dataRecorder2(sample2)
            if packages2:
                for package in packages2:
                    self.q_in2.put(package)
                    counter2 += 1
            counter_all += 1

            if keyboard.is_pressed("/"):      
                # 生产结束标志位
                self.q_in1.put("finish")
                self.q_in2.put("finish")
                print(f"daq1打包了{counter1}次数据, daq2打包了{counter2}，循环了{counter_all}次")

                self.daq1.unsubscribe(self.path1)
                self.daq2.unsubscribe(self.path2)


class DataConsumer(Thread):
    def __init__(
        self, 
        dataProcesser: Processer, 
        q_out1: Queue, 
        q_out2: Queue, 
        save_folder, 
        ser=None, 
        plot_online=False,
        ):
        super(DataConsumer, self).__init__()
        self.dataProcesser = dataProcesser
        self.q_out1 = q_out1
        self.q_out2 = q_out2
        self.save_folder = save_folder
        self.plot_online = plot_online
        self.ser = ser

    def run(self):
        package_counter = 0
        signal_counter = 0
        start_t = time.time()
        while True:
            
            packageGet1 = self.q_out1.get()
            if packageGet1 == "finish":
                break
            if time.time() - start_t <= 3:
                print(time.time())
                continue
            packageGet2 = self.q_out2.get()

            signal1 = packageGet1["signal"]
            timestamp1 = packageGet1["timestamp"]
            
            signal2 = packageGet2["signal"]
            timestamp2 = packageGet2["timestamp"]
            
            label1, borders1, pred_prob1 = self.dataProcesser(np.asarray(signal1))
            label2, borders2, pred_prob2 = self.dataProcesser(np.asarray(signal2))
            # processing_t = time.time() - start_t

            if label1 == 1 and label2 == 1:
                flag = "pulse"
                signal_counter += 1
                # decision making
                decision = self.dataProcesser.decision((signal1, borders1), (signal2, borders2))
                save_and_plot_doubleChannel(
                    save_folder=self.save_folder, 
                    raw_signal1=signal1, raw_signal2=signal2,
                    borders1=borders1, borders2=borders2,
                    timestamp1=timestamp1, timestamp2=timestamp2,
                    count=signal_counter, 
                    plot_online=self.plot_online,
                    )

                print(f"Processer made a decision of {decision}, It's {flag}")
            # print("Processing takes {:.5f} seconds".format(processing_t))

            # 向串口写入0-2
            # 0 -> 噪声   状态0
            # 1 -> 类别1  状态1
            # 2 -> 类别2  状态2
                if self.ser:
                    self.ser.write(str(decision).encode('utf-8'))

            package_counter += 1
        
        print(f"共接收到{package_counter}次数据包, {signal_counter}次信号")
        if not self.plot_online:
            post_plot(json_folder=self.save_folder)


# ------ test -------#
class Producer(Thread):
    def __init__(self, q_in: Queue):
        # Thread.__init__(self)
        super().__init__()
        self.q_in = q_in
        
    def run(self):
        print("produce start")
        i = 0
        while True:
            i += 1
            self.q_in.put(i)

            if keyboard.is_pressed("/"):
                self.q_in.put("finish")
                break
        print("生产者结束")
        

class Consumer(Thread):
    def __init__(self, q_out: Queue):
        super(Consumer, self).__init__()
        self.q_out = q_out
    
    def run(self):
        print("consume start")
        while True:
            data = self.q_out.get()
            if data == "finish":
                break
        print("消费者结束")


if __name__ == "__main__":
    q = Queue()
    producer = Producer(q)
    consumer = Consumer(q)
    producer.start()
    consumer.start()

