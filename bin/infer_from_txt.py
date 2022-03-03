# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   infer_from_txt.py
@Time    :   2022/03/01 16:54:56
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   从原始txt文件中读取数据并推理信号，作后处理 (txt数据和对应的peka json文件 应在同一目录下且同名)
-------------------------
'''
import os, sys
import shutil
from random import random
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from tools.infer import PeakDetector
from utils.utils import read_from_txt


def full_signal_infer(full_signal: list, model: PeakDetector):
    signalLen = len(full_signal)
    startPos = 0
    sliceLen = 200
    peakCollections = {
        "peaks": [],
        "ppVals": [],
        "travelTimes": [],
    }
    
    # 切片推理
    pbar = tqdm(total=signalLen)
    while startPos <= signalLen - sliceLen:
        signalSlice = np.array(full_signal[startPos: startPos + sliceLen], dtype=np.float32)
        label, borders, _ = model(signalSlice)
        if label == 1:
            # 抽取信息
            info = parsePeaks(signalSlice.tolist(), borders)
            if info:
                travelTime = info[0]
                ppVal = info[1]
                peakCollections["peaks"].append(signalSlice.tolist())
                peakCollections["ppVals"].append(ppVal)
                peakCollections["travelTimes"].append(travelTime)

        startPos += sliceLen
        pbar.update(sliceLen)
    pbar.close()

    return peakCollections


def parsePeaks(peak: list, borders: list):
    """ 从peak中抽取pp值和travelTime的信息
        暂时只考虑一个峰值的情况

    Args:
        peak (list): peak signal slice
        borders (_type_): border of the peak
    """
    sliceLen = len(peak)
    borderNum = len(borders) if borders else 0
    
    if borderNum == 1: # 只有一个峰值的情况
        begin, end = borders[0][0], borders[0][1]
        # maxVal = max(peak)
        # minVal = min(peak)
        # begin = peak.index(maxVal)
        # end = peak.index(minVal)
        maxVal = max(peak[begin:end])
        minVal = min(peak[begin:end])
        travelPoints = abs(begin - end)
        travelTime = travelPoints * (1 / 1674) * 1000
        ppVal = abs(maxVal - minVal)
        if ppVal > 0.0012:
            return None
        else:
            return (travelTime, ppVal)
    elif borderNum >= 2 : # 多于一个峰值的情况: 返回信号最长的那个
        maxBorder = []
        maxBorderLen = 0
        for border in borders:
            borderLen = abs(border[0] - border[1])
            if borderLen > maxBorderLen:
                maxBorderLen = borderLen
                maxBorder = border

        begin, end = maxBorder[0], maxBorder[1]
        maxVal = max(peak[begin:end])
        minVal = min(peak[begin:end])
        travelPoints = abs(begin - end)
        travelTime = travelPoints * (1 / 1674) * 1000
        ppVal = abs(maxVal - minVal)
        if ppVal > 0.0012:
            return None
        else:
            return (travelTime, ppVal)
    elif not borders:
        return None


def peakJsonLoad(jsonFile):
    # 解析存下的peak json文件
    with open(jsonFile, 'r') as fp:
        peakCollections = json.load(fp)
    ppVals, travelTimes = peakCollections["ppVals"], peakCollections["travelTimes"]
    return ppVals, travelTimes


def plotTimePP(txt_dict):
    fig = plt.figure()
    grid_spec = GridSpec(4, 4)
    ax_scatter = fig.add_subplot(grid_spec[1:, 0:3]) # scatter axes
    ax_xhist = fig.add_subplot(grid_spec[0, 0:3]) # x hist axes
    ax_yhist = fig.add_subplot(grid_spec[1:, 3]) # y hist axes
    
    # 从peak json中读取信息并作图
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    i = 0
    for label, txt_file in txt_dict.items():
        if not txt_file:
            continue
        peak_json_file = os.path.splitext(txt_file)[0] + ".json"
        assert os.path.exists(peak_json_file), "json file does not exist: {}".format(peak_json_file)
        ppVals, travelsTimes = peakJsonLoad(peak_json_file)
        print("{} has {} peaks".format(label, len(ppVals)))
        # color = (random(), random(), random())
        color = colors[i]
        ax_scatter.scatter(travelsTimes, ppVals, label=label, color=color, s=2, )  # scatter
        _, xhist_bins, _ = ax_xhist.hist(travelsTimes, bins=200, color=color, alpha=0.5)  # hist travel time
        ax_xhist.set_xticks([])
        _, yhist_bins, _ = ax_yhist.hist(ppVals, bins=200, orientation='horizontal', color=color, alpha=0.5)  # hist ppvals
        ax_yhist.set_yticks([])
        # ax_xhist.plot(xhist_bins, )
        i += 1
    
    ax_scatter.set_xlabel("travel time (ms)")
    ax_scatter.set_ylabel("peak2peak value (V)")
    ax_scatter.legend()

    ax_xhist.set_ylabel("count")
    ax_yhist.set_xlabel("count")

    fig.suptitle("travel time and peak2peak value")
    fig_name = label[:3] + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
    fig.savefig(os.path.join("../data/", fig_name))
        

if __name__ == "__main__":

    # 模型权重文件
    classifierModelPath = r"../data\weights\2022_0303\Classifier"
    segmentatorModelPath = r"../data\weights\2022_0303\Segmentator"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # 绘图标签
    cell_type = "MCF7"
    label1 = cell_type + "_Antibody_Coated"
    label2 = cell_type + "_LPA"
    label3 = cell_type + "_Unprocessed"

    # txt 路径列表
    txt_path_list = {
        # label1: r"../data/raw_txt_data/MutiChannel_MDA_ANTIBODY_COATING_2022_02_27/1_05psi_noGravity_meas_plotter_20220227_185626.txt",
        # label2: r"../data\raw_txt_data\MultiChannel_MDA_LPA_2022_02_16\4B_Middle_05psi_meas_plotter_20220216_173345.txt",
        # label3: r"../data\raw_txt_data\MultiChannel_MDA_BASE_2022_03_02/3_sparse_good_05psi_meas_plotter_20220302_165416.txt",

        label1: r"../data/raw_txt_data/MultiChannel_MCF7_ANTIBODY_COATING_20220227/6_withGravity_02psi_meas_plotter_20220227_181401.txt",
        label2: r"../data\raw_txt_data\MultiChannel_MCF7_LPA_20220215\4B_Middle_02psi_meas_plotter_20220215_162237.txt",
        label3: r"../data\raw_txt_data\MultiChannel_MCF7_BASE_20220302/5_dense_good_05psi_meas_plotter_20220302_152537.txt",
    }

    peakDetector = PeakDetector(
        classifier_weight=classifierModelPath,
        segmentator_weight=segmentatorModelPath,
        device=device,
    )

    # 推理txt文件，将解析后的信号时间峰值保存到json
    for txt_file in txt_path_list.values():
        if not txt_file:
            continue
        # 检查是否已经推理过
        peak_json_file = os.path.splitext(txt_file)[0] + ".json"
        if os.path.exists(peak_json_file):
            # os.remove(peak_json_file)
            continue
        signal_data, _ = read_from_txt(txt_file)
        peakCollections = full_signal_infer(signal_data, peakDetector)
        with open(peak_json_file, 'w', encoding='utf-8') as pj:
            json.dump(peakCollections, pj, indent=4)
        
    plotTimePP(txt_path_list)
        

