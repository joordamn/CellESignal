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
from random import random
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def parsePeaks(peak: list, borders):
    """ 从peak中抽取pp值和travelTime的信息
        暂时只考虑一个峰值的情况

    Args:
        peak (list): peak signal slice
        borders (_type_): border of the peak
    """
    sliceLen = len(peak)
    borderNum = len(borders) if borders else 0
    
    if borderNum == 1 or not borders: # 只有一个峰值的情况
        maxVal = max(peak)
        minVal = min(peak)
        maxInd = peak.index(maxVal)
        minInd = peak.index(minVal)
        travelPoints = abs(maxInd - minInd)
        travelTime = travelPoints * 1 / 1667
        ppVal = abs(maxVal - minVal)
        return (travelTime, ppVal)
    elif borderNum >= 2 : # 多于一个峰值的情况:
        return None


def peakJsonLoad(jsonFile):
    # 解析存下的peak json文件
    with open(jsonFile, 'r') as fp:
        peakCollections = json.load(fp)
    ppVals, travelTimes = peakCollections["ppVals"], peakCollections["travelTimes"]
    return ppVals, travelTimes


def plotTimePP(txt_dict):
    # 从peak json中读取信息并作图
    for label, txt_file in txt_dict.items():
        if not txt_file:
            continue
        peak_json_file = os.path.splitext(txt_file)[0] + ".json"
        assert os.path.exists(peak_json_file), "json file does not exist: {}".format(peak_json_file)
        ppVals, travelsTimes = peakJsonLoad(peak_json_file)
        print("{} has {} peaks".format(label, len(ppVals)))
        color = (random(), random(), random())
        plt.scatter(travelsTimes, ppVals, label=label, edgecolors= 'white', color=color, s=10)
    plt.xlabel("travel time (ms)")
    plt.ylabel("peak2peak value (V)")
    plt.legend()
    plt.title("travel time and peak2peak value")
    plt.savefig("../data/1.png")
        

if __name__ == "__main__":

    # 模型权重文件
    classifierModelPath = r"../data\weights\2022_0302\Classifier"
    segmentatorModelPath = r"../data\weights\2022_0302\Segmentator"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 绘图标签
    label1 = "MDA_Antibody_Coated"
    label2 = "MDA_LPA"
    label3 = "MDA_Unprocessed"

    # txt 路径列表
    txt_path_list = {
        label1: r"../data/raw_txt_data/MutiChannel_MDA_ANTIBODY_COATING_2022_02_27/1_05psi_noGravity_meas_plotter_20220227_185626.txt",
        label2: r"../data\raw_txt_data\MultiChannel_MDA_LPA_2022_02_16\4B_Middle_05psi_meas_plotter_20220216_173345.txt",
        label3: r"../data\raw_txt_data\MultiChannel_MDA_BASE_2022_03_02/3_sparse_good_05psi_meas_plotter_20220302_165416.txt",
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
        if not os.path.exists(peak_json_file):
            signal_data, _ = read_from_txt(txt_file)
            peakCollections = full_signal_infer(signal_data, peakDetector)
            with open(peak_json_file, 'w', encoding='utf-8') as pj:
                json.dump(peakCollections, pj, indent=4)
        
    plotTimePP(txt_path_list)
        

