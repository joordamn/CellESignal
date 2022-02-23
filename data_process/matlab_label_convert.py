# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   matlab_label_convert.py
@Time    :   2022/01/24 15:36:12
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   此脚本用于转换matlab导出的标签txt
-------------------------
'''

import os, sys, shutil
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.chdir(sys.path[-1])

from tqdm import tqdm
import json
import random
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import read_from_txt


def parse(root_folder, signal_save_path, noise_save_path, noise_num_ratio=2, split_len=200, plot=True):
    """在txt文件的根目录下读取txt文件，并生成对应的json文件和图
    """

    label = {
            "code": "",
            "label": 1,
            "number of peaks": 1,
            "peaks' labels": [],
            "borders": [],
            "description": "",
            "rt":[],
            "scan": [],
            "intensity": [],
            "mz": [],
        }

    for item in os.listdir(root_folder):
        if item.endswith("txt"):
            if item.startswith("0"):
                # 读取信号数据
                signal_data, _ = read_from_txt(os.path.join(root_folder, item))
            elif item.startswith("r"):
                # 读取border数据
                border_data, _ = read_from_txt(os.path.join(root_folder, item))
    # 提取信号逐段分割
    counter = 0
    if plot:
        figure = plt.figure()

    rt_total = []
    for i, border in tqdm(enumerate(border_data)):
        label["peaks' labels"] = []
        label["borders"] = []
        
        begin, end = border
        border_len = end - begin

        if border_len >= int(split_len * 0.9):
            continue
        
        # 截取数据
        pad_len = random.randint(2, int(split_len - border_len))
        rt_start = begin - pad_len
        rt_end = rt_start + split_len
        data_slice = signal_data[rt_start:rt_end]
        rt_total.append([rt_start, rt_end])

        # 判断前两个后两个border是否包含
        border_contains = []
        if i >= 2 and i <= len(border_data) - 3:
            borders = border_data[i-2], border_data[i-1], border, border_data[i+2], border_data[i+1]
            for b in borders:
                if rt_start <= b[0] <= rt_end or rt_start <= b[1] <= rt_end:
                    _begin = max(0, b[0] - rt_start)
                    _end = min(split_len - 1, b[1] - rt_start)
                    border_contains.append([int(_begin), int(_end)])
                    label["peaks' labels"].append([0])
        else:
            border_contains.append([pad_len, min(split_len-1, pad_len + border_len)])
            label["peaks' labels"].append([0])

        # 改写json内容
        json_file_name = "peak_sample_" + str(counter).zfill(4)
        json_file = os.path.join(signal_save_path, json_file_name) + '.json'

        label["code"] = json_file_name
        label["number of peaks"] = len(border_contains)
        label["borders"] = border_contains
        label["intensity"] = data_slice
        label["rt"] = [rt_start, rt_end]
        label["mz"] = data_slice
        with open(json_file, mode="w", encoding="utf-8") as jf:
            json.dump(label, jf)
        
        # plot
        if plot:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.plot(data_slice)
            for i, border in enumerate(label['borders']):
                begin, end = border
                ax.fill_between(range(begin, end + 1), y1=data_slice[begin:end + 1], y2=min(data_slice), alpha=0.5,
                                label=f"peak NO: {i}, borders={begin}-{end}")
            ax.legend(loc='best')
            fig_save_path = signal_save_path + "/fig/"
            if not os.path.exists(fig_save_path):
                os.makedirs(fig_save_path)
            figure.savefig(fig_save_path + "peak_sample_" + str(counter).zfill(4) + ".jpg")
            plt.clf()
        
        counter += 1

    print("信号生成完成")

    # ---------生成噪声----------- #

    # 随机生成10000个噪声起点
    noise_locs = np.random.randint(100, len(signal_data)*0.5, (10000,)).tolist()
    num_of_noise = noise_num_ratio * len(border_data) # 噪声数量
    # 根据噪声位置和border位置，筛选合适的噪声起点
    filtered_noise_locs = []
    for i, noise_loc in enumerate(noise_locs):
        count = 0
        for border in border_data:
            begin, end = border[0], border[1]
            if noise_loc <= begin - 2 * split_len or noise_loc >= end + 2 * split_len:
                count += 1
        if count >= len(border_data):
            filtered_noise_locs.append(noise_loc)
    print("filtered noise has {}".format(len(filtered_noise_locs)))
    assert len(filtered_noise_locs) >= num_of_noise, "filtered noise num less than {0}".format(num_of_noise)
    final_noise_locs = filtered_noise_locs[:num_of_noise]
    # 截取噪声数据
    for i, loc in tqdm(enumerate(final_noise_locs)):
        noise_slice = signal_data[loc:loc + split_len]

        # 改写json内容
        json_file_name = "nois_sample_" + str(i).zfill(4)
        json_file = os.path.join(noise_save_path, json_file_name) + '.json'
        
        label["borders"] = []
        label["label"] = 0
        label["number of peaks"] = 0
        label["peaks' labels"] = []
        label["code"] = json_file_name
        label["intensity"] = noise_slice
        label["rt"] = [loc, loc + split_len]
        label["mz"] = noise_slice
        with open(json_file, mode="w", encoding="utf-8") as jf:
            json.dump(label, jf)
        
        # plot
        if plot:
            figure.clear()
            ax = figure.add_subplot(111)
            random_signal = rt_total[random.randint(0, len(rt_total)-1)][0]
            ax.plot(signal_data[random_signal:random_signal+split_len])
            ax.plot(noise_slice)
            ax.set_title(label["code"])
            fig_save_path = noise_save_path + "/fig/"
            if not os.path.exists(fig_save_path):
                os.makedirs(fig_save_path)
            figure.savefig(fig_save_path + "nois_sample_" + str(i).zfill(4) + ".jpg")
            plt.clf()

    print("噪声生成完成")
            


if __name__ == "__main__":
    root_folders = [
        "../data/data_collection_20220115/txt_data/01",
        "../data/data_collection_20220115/txt_data/02",
        "../data/data_collection_20220115/txt_data/03",
    ]

    for root_folder in root_folders:

        peak_save_path = root_folder + "/peak_data/"
        noise_save_path = root_folder + "/noise_data/"
        
        try:
            shutil.rmtree(peak_save_path)
            shutil.rmtree(noise_save_path)
        except:
            pass
        
        if not os.path.exists(peak_save_path):
            os.makedirs(peak_save_path)
            os.makedirs(noise_save_path)
        
        parse(root_folder, peak_save_path, noise_save_path,plot=True)