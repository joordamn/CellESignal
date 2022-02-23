# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   daq_session_creator.py
@Time    :   2022/01/05 22:21:51
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   创建daq的api session
-------------------------
'''

import zhinst.utils
import time

device_id = "dev3051"
server_host = "localhost"
server_port = 8004
api_level = 6

# out_mixer_channel = zhinst.utils.default_output_mixer_channel(props) # 混频通道
out_channel = 0                     # 信号输出通道
in_channel = 0                      # 信号输入通道
demod_index = 0                     # 解调器序号
demod_rate = 1600                   # 解调采样频率
time_constant = 0.000540940626      # 低通滤波器 时间常数 此值为3rd 0.0027->30dB  0.00162282188-> 50.41dB 0.000819607008->99dB 0.000540940626->150dB
order = 3                           # 滤波器阶数
frequency = 303e3                   # 输出信号频率
amplitude = 50                       # 输出信号幅值

def createSession(
    device_id=device_id, 
    server_host=server_host, 
    server_port=server_port, 
    api_level=api_level,
    out_channel=out_channel,
    in_channel=in_channel,
    demod_index=demod_index,
    demod_rate=demod_rate,
    time_constant=time_constant,
    order=order,
    frequency=frequency,
    amplitude=amplitude,
    ):

    (daq, device, props) = zhinst.utils.create_api_session(
        device_serial=device_id,
        api_level=api_level,
        server_host=server_host,
        server_port=server_port,
    )
    
    zhinst.utils.disable_everything(daq=daq, device=device_id)

    exp_setting = [

    [f"/{device}/sigins/{in_channel}/ac", 0],
    [f"/{device}/sigins/{in_channel}/range", 1],#2 * amplitude],
    [f"/{device}/sigins/{in_channel}/diff", 1],

    [f"/{device}/sigouts/{out_channel}/on", 1],
    [f"/{device}/sigouts/{out_channel}/enables/1", 1],
    [f"/{device}/sigouts/{out_channel}/amplitudes", 2 * amplitude],
    [f"/{device}/sigouts/{out_channel}/range", 10],
    [f"/{device}/sigouts/{out_channel}/imp50", 0],
    [f"/{device}/oscs/{out_channel}/freq", frequency],

    [f"/{device}/demods/{demod_index}/enable", 1],
    [f"/{device}/demods/{demod_index}/rate", demod_rate],
    [f"/{device}/demods/{demod_index}/order", order],
    [f"/{device}/demods/{demod_index}/timeconstant", time_constant],

    ]
    
    daq.set(exp_setting)
    daq.unsubscribe("*")

    path = f"/{device}/demods/{demod_index}/sample"

    time.sleep(1)

    return daq, device, path
