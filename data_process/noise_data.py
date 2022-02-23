# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   noise_data.py
@Time    :   2021/12/28 17:58:18
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   此脚本用于生成和峰值点没有重叠的噪声样本
-------------------------
'''
import json
from re import T
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os, sys, shutil
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.chdir(sys.path[-1])


def noise_split(data:np.ndarray, loc_list:list, save_path:str, split_len=50, plot=False, num_of_noise=50):
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

    # 随机生成10000个噪声中心点
    noise_locs = np.random.randint(100, len(data), (10000,)).tolist()

    # 噪声中心点和峰值中心点作差，筛选出合适的噪声中心点
    filtered_noise_locs = []
    for i, noise_loc in enumerate(noise_locs):
        count = 0
        for peak_loc in loc_list:
            if abs(peak_loc - noise_loc) > 2 * split_len:
                count += 1
        if count >= len(loc_list):
            filtered_noise_locs.append(noise_loc)
    
    print("filtered noise has {}".format(len(filtered_noise_locs)))
    assert len(filtered_noise_locs) >= num_of_noise, "filtered noise num less than {0}".format(num_of_noise)
    final_noise_locs = filtered_noise_locs[:num_of_noise]

    # 截取噪声数据
    for i, loc in tqdm(enumerate(final_noise_locs)):
        noise_slice = data[loc - split_len: loc + split_len].tolist()

        # 改写json内容
        json_save_name = save_path + "nois_sample_" + str(i).zfill(4)
        json_file = json_save_name + ".json"
        
        label["code"] = "data slice NO_" + str(i).zfill(4)
        label["intensity"] = noise_slice
        label["rt"] = [loc - split_len, loc + split_len]
        label["mz"] = noise_slice
        with open(json_file, mode="w", encoding="utf-8") as jf:
            json.dump(label, jf)
        
        # plot
        if plot:
            plt.figure()
            plt.plot(noise_slice)
            fig_save_path = save_path + "/fig/"
            if not os.path.exists(fig_save_path):
                os.makedirs(fig_save_path)
            plt.savefig(fig_save_path + "nois_sample_" + str(i).zfill(4) + ".jpg")
            plt.close("all")


if __name__ == "__main__":

    raw_data_file = "./rawData.csv" # 原始数据
    raw_peak_loc_file = "./raw_data_loc.txt" # 原始数据的峰值点坐标
    save_path = "./noise_data/"
    split_len = 50
    num_of_noise = 500

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

    noise_split(data=raw_data, loc_list=loc, save_path=save_path, split_len=split_len, plot=True, num_of_noise=num_of_noise)