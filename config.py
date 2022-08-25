# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   config.py
@Time    :   2022/08/15 15:07:26
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   config文件
-------------------------
'''


from easydict import EasyDict
import torch

cfg = EasyDict()

# cpu or gpu
cfg.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# production queue length
cfg.queueLen = 0

# model weight path
cfg.classifierModelPath = r"../data\weights\2022_0303\Classifier"
cfg.segmentatorModelPath = r"../data\weights\2022_0303\Segmentator"

# exp 实验相关参数设置
cfg.exp_root = "./exp"
cfg.exp_mode = "liveordead" # 实验种类 "liveordead" / "stiffness"
cfg.sorting_gate = {
    "ppVal": [[0.0002, 0.008],[0, 0.0002]],  # 活细胞峰值范围，死细胞峰值范围
    "travel_time": [[40, 60],[70, 80]], # 活细胞速度范围，死细胞速度范围 ms
}

# serial port
cfg.serialPort = "COM4"
cfg.baudRate = 115200

# ---------------------------------------------------------------#
# daq module params 
# common params
cfg.server_host = "localhost"
cfg.api_level = 6
cfg.out_channel = 0                     # 信号输出通道
cfg.in_channel = 0                      # 信号输入通道
cfg.demod_index = 0                     # 解调器序号
cfg.demod_rate = 1600                   # 解调采样频率
cfg.time_constant = 0.000540940626      # 低通滤波器 时间常数 此值为3rd 0.0027->30dB  0.00162282188-> 50.41dB 0.000819607008->99dB 0.000540940626->150dB
cfg.order = 3                           # 滤波器阶数
cfg.frequency = 303e3                   # 输出信号频率
cfg.amplitude = 50                      # 输出信号幅值
# daq 1
cfg.server_1_port = 8004
cfg.device_1_id = "dev3051"
cfg.frequency_1 = 303e3
# daq 2
cfg.server_2_port = 8004
cfg.device_2_id = "dev5234"
cfg.frequency_2 = 289e3

