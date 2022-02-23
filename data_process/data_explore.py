# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   data_explore.ipynb
@Time    :   2021/12/28 17:58:18
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   此脚本用于将原始数据及导出的峰值坐标进行裁剪，输出json格式，用于后续打标训练用
-------------------------
'''
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os, sys, shutil

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.chdir(sys.path[-1])


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
    for i, loc in tqdm(enumerate(loc_list[1000:1500])):
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

    raw_data_file = "./rawData.csv" # 原始数据
    raw_peak_loc_file = "./raw_data_loc.txt" # 原始数据的峰值点坐标
    save_path = "./peak_data/"
    split_len = 50

    raw_data = np.genfromtxt(raw_data_file, delimiter=",")

    with open(raw_peak_loc_file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        loc = []
        for line in lines:
            loc.append(int(line))

    try:
        shutil.rmtree(save_path)
    except:
        pass

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_split(data=raw_data, loc_list=loc, save_path=save_path, split_len=split_len, plot=True)