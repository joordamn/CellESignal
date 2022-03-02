# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   infer_from_txt.py
@Time    :   2022/03/01 16:54:56
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   从原始txt文件中读取数据并推理信号，作后处理
-------------------------
'''
import os, sys
import torch
import numpy as np
from tqdm import tqdm
import json

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
            info = parsePeaks(signalSlice, borders)
            if info:
                travelTime = info[0]
                ppVal = info[1]
                peakCollections["peaks"].append(signalSlice.tolist())
                peakCollections["ppVals"].append(float(ppVal))
                peakCollections["travelTimes"].append(float(travelTime))

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
    borderNum = len(borders)
    peak = np.array(peak)
    
    if borderNum == 1 or not borders: # 只有一个峰值的情况
        maxVal = np.max(peak)
        minVal = np.min(peak)
        maxInd = np.argmax(peak)
        minInd = np.argmin(peak)
        travelPoints = abs(maxInd - minInd)
        travelTime = travelPoints * 1 / 1667
        ppVal = abs(maxVal - minVal)
        return (travelTime, ppVal)
    elif borderNum >= 2 : # 多于一个峰值的情况:
        return None
        

if __name__ == "__main__":

    # 模型权重文件
    classifierModelPath = r"../data\weights\best\2022_0301\Classifier"
    segmentatorModelPath = r"../data\weights\best\2022_0301\Segmentator"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # txt 路径列表
    txt_path_list = [
        "../data/raw_txt_data/MutiChannel_MDA_ANTIBODY_COATING_2022_02_27/1_05psi_noGravity_meas_plotter_20220227_185626.txt",
    ]

    peakDetector = PeakDetector(
        classifier_weight=classifierModelPath,
        segmentator_weight=segmentatorModelPath,
        device=device,
    )

    for txt_file in txt_path_list:
        peak_json_file = os.path.splitext(txt_file)[0] + ".json"
        # 检查是否已经推理过
        if not os.path.exists(peak_json_file):
            signal_data, _ = read_from_txt(txt_file)
            peakCollections = full_signal_infer(signal_data, peakDetector)
            with open(peak_json_file, 'w', encoding='utf-8') as pj:
                json.dumps(peakCollections, pj, indent=4)
        
        # 从peak json中读取信息并作图
        
