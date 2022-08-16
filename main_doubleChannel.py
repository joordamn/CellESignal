# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   doubleChannel.py
@Time    :   2022/08/15 14:59:15
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   用于双通道检测，开两个daq，一个生产者线程同时包含两个daq对象
-------------------------
'''


from queue import Queue
from serial import Serial

from src.processer import Processer
from src.production import DataProducer, DataConsumer
from src.recorder import Recorder
from utils.daqSessionCreator import DaqSession
from utils.utils import create_folder
from config import cfg


def connectAmplifier(cfg):
    daqSession_1 = DaqSession(cfg, daq_num=1)
    daqSession_2 = DaqSession(cfg, daq_num=2)
    print("-"*10 + "成功连接放大器" + "-"*10)
    return daqSession_1, daqSession_2


def connectArduino(cfg):
    try:
        ser = Serial(cfg.serialPort, cfg.baudRate, timeout=0.5)
        print("-"*10 + "成功连接Arduino" + "-"*10)
    except Exception as e:
        print("-"*10 + "未能连接到Arduino" + "-"*10 + "\n")
        print(e)
        ser = None
    return ser


def buildThread(cfg):
    # 连接arduino和放大器
    infer_result_save_folder = create_folder(cfg.exp_root)
    ser = connectArduino(cfg)
    daqSession1, daqSession2 = connectAmplifier(cfg)

    # 初始化recorder和processor
    recorder1 = Recorder(200) # 打包200长度的数据
    recorder2 = Recorder(200)
    processer = Processer(
        classifier_weight=cfg.classifierModelPath,
        segmentator_weight=cfg.segmentatorModelPath,
        device=cfg.DEVICE
    )
    q1 = Queue(cfg.queueLen)
    q2 = Queue(cfg.queueLen)
    
    # 初始化生产者和检测器线程
    producer = DataProducer(
        dataRecorder1=recorder1,
        dataRecorder2=recorder2, 
        daqSession1=daqSession1, 
        daqSession2=daqSession2, 
        q_in1=q1,
        q_in2=q2,
        )
    consumer = DataConsumer(
        dataProcesser=processer, 
        q_out1=q1, 
        q_out2=q2,
        save_folder=infer_result_save_folder, 
        ser=ser,
        plot_online=True,
        )

    print("-"*10 + "模型加载完毕, 线程已开启" + "-"*10)
    return producer, consumer


def run(cfg):
    producer, consumer = buildThread(cfg)
    producer.start()
    consumer.start()
    
    print("-"*10 + "程序正在运行" + "-"*10 + "\n")


if __name__ == "__main__":
    
    run(cfg)
