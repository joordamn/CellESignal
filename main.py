# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   main.py
@Time    :   2022/01/05 22:02:48
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   细胞电信号检测、决策系统入口
-------------------------
'''

from queue import Queue
import torch
from threading import Thread
import numpy as np
from serial import Serial

from src.processer import Processer
from src.production import dataProduction, dataProcess
from src.recorder import Recorder
from utils.daqSessionCreator import DaqSession
from utils.utils import create_folder, post_plot
from config import cfg


if __name__ == "__main__":
    #-------------------参数设置----------------#

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifierModelPath = r"../data\weights\2022_0303\Classifier"
    segmentatorModelPath = r"../data\weights\2022_0303\Segmentator"

    serialPort = "COM3"
    baudRate = 115200

    queueLen = 0

    exp_root = "./exp"
    infer_result_save_folder = create_folder(exp_root)

    #-------------------连接设备----------------#

    daq_session = DaqSession(cfg)
    print("-"*10 + "成功连接设备" + "-"*10)


    #-------------------初始化Arduino连接-------------------#

    try:
        ser = Serial(serialPort, baudRate, timeout=0.5)
        print("-"*10 + "成功连接" + "-"*10)
    except Exception as e:
        print("-"*10 + "未能连接到Arduino" + "-"*10 + "\n")
        print(e)
        ser = None

    #-------------------初始化Recorder和Processor----------#

    recorder = Recorder(200) # 默认打包100长度的数据
    processer = Processer(
        classifier_weight=classifierModelPath,
        segmentator_weight=segmentatorModelPath,
        device=DEVICE
    )
    
    #------------------开启生产者和检测器线程----------------#
    q = Queue(queueLen)
    t_producer = Thread(target=dataProduction, args=(recorder, daq_session.daq, q, daq_session.path))
    t_consumer = Thread(target=dataProcess, args=(processer, q, infer_result_save_folder, ser))

    print("-"*10 + "模型加载完毕, 线程已开启" + "-"*10)

    #------------------开始运行--------------------------------#

    t_producer.start()
    t_consumer.start()

    print("-"*10 + "程序正在运行" + "-"*10 + "\n")






    
