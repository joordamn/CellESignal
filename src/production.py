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

from queue import Queue
import numpy as np
import time

from src.recorder import Recorder
from src.processer import Processer
from utils.utils import save_and_plot, post_plot


def dataProduction(recorderIns: Recorder, daq, q_in: Queue, device, path, t_end=500):
    """生产者线程函数

    """

    counter = 0
    counter2 = 0
    curr_time = time.time()
    daq.sync()
    daq.subscribe(path)
    time.sleep(0.05)
    while time.time() - curr_time < t_end: # or len(recorderIns.streamData["timestamp"]) > recorderIns._overlap:
    # while True:

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
        decision, label, borders, pred_prob = processer(np.asarray(signal))
        processing_t = time.time() - start_t
        if label == 0:
            flag = "noise"
        elif label == 1:
            flag = "pulse"
            signal_counter += 1
            save_and_plot(save_folder=save_folder, raw_signal=signal, borders=borders, pred_prob=pred_prob,timestamp=timestamp, count=signal_counter, plot_online=plot_online)

            print(f"Processer made a decision of {decision}, It's {flag}")
        # print("Processing takes {:.5f} seconds".format(processing_t))

        # 向串口写入0-2
        # 0 -> 噪声   状态0
        # 1 -> 类别1  状态1
        # 2 -> 类别2  状态2
            if ser:
                ser.write(str(decision).encode('utf-8'))

        package_counter += 1
    print(f"共接收到{package_counter}次数据包, {signal_counter}次信号")
    if not plot_online:
        post_plot(json_folder=save_folder)