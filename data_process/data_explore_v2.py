# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   data_explore.ipynb
@Time    :   2022/01/20 14:11
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   此脚本用于
                        1) 读取原始txt数据
                        2) 寻找峰值点及其坐标
                        3) 将原始数据及导出的峰值坐标进行裁剪，输出json格式，用于后续打标训练用
-------------------------
'''
import os, sys, shutil
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.chdir(sys.path[-1])

import json
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.signal import find_peaks

from utils.utils import read_from_txt


def peak_finding(signal_data: np.ndarray):
    """找出峰值点及其坐标
    """
    loc, _ = find_peaks(signal_data, prominence=0.00005)
    loc = np.squeeze(loc).tolist()
    return loc


def data_split(data:np.ndarray, loc_list:list, save_path:str, split_len=150, plot=False):
    """根据loc的位置去前后截取raw_data的数据
    """
    label = {
        "code": "",
        "label": 0,
        "number of peaks": 0,
        "peaks' labels": [],
        "borders": [],
        "description": "",
        "rt":[],
        "scan": [],
        "intensity": [],
        "mz": [],
    }
    for i, loc in tqdm(enumerate(loc_list)):
        # 截取数据
        # 将loc的位置随机前后位移 使得峰值点不在数据切片的中心
        loc += random.randint(int(-1 * 1/3 * split_len), int(1/3 * split_len))
        data_slice = data[loc - split_len: loc + split_len].tolist()

        # 改写json内容
        json_save_name = save_path + "peak_sample_" + str(i).zfill(4)
        json_file = json_save_name + ".json"
        
        label["code"] = "data slice NO_" + str(i).zfill(4)
        label["intensity"] = data_slice
        label["rt"] = [loc - split_len, loc + split_len]
        label["mz"] = data_slice
        with open(json_file, mode="w", encoding="utf-8") as jf:
            json.dump(label, jf)
        
        # plot
        if plot:
            plt.figure()
            plt.plot(data_slice)
            fig_save_path = save_path + "/fig/"
            if not os.path.exists(fig_save_path):
                os.makedirs(fig_save_path)
            plt.savefig(fig_save_path + "peak_sample_" + str(i).zfill(4) + ".jpg")
            plt.close("all")


if __name__ == "__main__":

    raw_data_file = r"../data\data_collection_20220115\txt_data\150dB_1V_highspeed.txt" # 原始数据
    # raw_peak_loc_file = "./raw_data_loc.txt" # 原始数据的峰值点坐标
    save_path = r"../data/data_collection_20220115/peak_data_03/"
    split_len = 100

    raw_data, _ = read_from_txt(raw_data_file)
    raw_data = np.array(raw_data, dtype=np.float32)

    loc = peak_finding(raw_data)

    print(len(loc))
    # plt.figure()
    # plt.plot(loc, raw_data[loc], "xr")
    # plt.plot(raw_data)
    # plt.show()

    try:
        shutil.rmtree(save_path)
    except:
        pass

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_split(data=raw_data, loc_list=loc, save_path=save_path, split_len=split_len, plot=True)