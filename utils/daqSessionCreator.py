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

import os, sys
import zhinst.utils
import time
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)


class DaqSession:
    def __init__(self, cfg, daq_num=1):
        self.cfg = cfg
        self.daq_num = daq_num
        self.daq, self.device, self.path = self.createSession(self.cfg, self.daq_num)

    def createSession(
        self,
        cfg,
        daq_num = 1,
        ):
        # unique params
        device_id     = cfg.device_1_id 
        server_port   = cfg.server_1_port
        frequency     = cfg.frequency_1
        # common params
        server_host   = cfg.server_host
        api_level     = cfg.api_level
        out_channel   = cfg.out_channel
        in_channel    = cfg.in_channel
        demod_index   = cfg.demod_index
        demod_rate    = cfg.demod_rate
        time_constant = cfg.time_constant
        order         = cfg.order
        amplitude     = cfg.amplitude

        if daq_num == 2:
            device_id = cfg.device_2_id
            server_port = cfg.server_2_port
            frequency = cfg.frequency_2

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
