# -*- encoding: utf-8 -*-
"""
-------------------------
@File    :   infer.py
@Time    :   2021/12/31 14:23:11
@Author  :   Zhongning Jiang
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :
-------------------------
"""

import json, os, sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from scipy.interpolate import interp1d

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator
from utils.utils import read_from_txt


class PeakDetector:
    def __init__(self, classifier_weight, segmentator_weight, device, interpolate_length=256):
        self.device = device
        self.interpolate_length = interpolate_length

        # load classifier
        self.classifier = Classifier().to(self.device)
        self.classifier.load_state_dict(torch.load(classifier_weight, map_location=self.device))
        self.classifier.eval()
        # load segmentor
        self.segmentator = Segmentator().to(self.device)
        self.segmentator.load_state_dict(torch.load(segmentator_weight, map_location=self.device))
        self.segmentator.eval()

    def __call__(self, raw_data: np.ndarray):
        signal = self._preprocess(raw_data, interpolate=True, length=self.interpolate_length)
        # model inference
        class_output= self.classifier(signal)
        class_output = class_output.data.cpu().numpy()
        # get label
        label = np.argmax(class_output)
        # peak detected
        if label == 1:
            seg_output = self.segmentator(signal)
            seg_output = seg_output.data.sigmoid().cpu().numpy()
            borders = self._get_borders(seg_output[0, 0], interpolation_factor=len(signal[0, 0]) / len(raw_data))
            return 1, borders, seg_output[0, 0].tolist()
        # no peak detected
        else:
            return 0, None, None

    def _preprocess(self, signal, interpolate=True, length=None):
        """
        preprocess the input signal to keep the consistence with training process
        """
        if not length:
            length = self.interpolate_length
        if interpolate:
            interpolate = interp1d(np.arange(len(signal)), signal, kind='linear')
            signal = interpolate(np.arange(length) / (length - 1) * (len(signal) - 1))
        # normalize
        signal = torch.tensor(signal / np.max(signal), dtype=torch.float32, device=self.device)

        return signal.view(1, 1, -1)

    def _get_borders(self, pred_prob, threshold=0.5, interpolation_factor=1, minimum_peak_points=10):
        """ post process for the predicted probability and find the peaks
        """
        pred_mask = pred_prob > threshold  # threshold cut the prediction
        borders_roi = []
        begin = 0 if pred_mask[0] else -1  # init peak begin point
        peak_wide = 1 if pred_mask[0] else 0  # init peak wide
        number_of_peaks = 0
        for n in range(len(pred_mask) - 1):  # loop the mask and analyze the peak
            if pred_mask[n + 1] and not pred_mask[n]:  # case1: peak begins
                begin = n + 1
                peak_wide = 1
            elif pred_mask[n + 1] and begin != -1:  # case2: peak continues
                peak_wide += 1
            elif not pred_mask[n + 1] and begin != -1:  # case3: peak ends
                if peak_wide / interpolation_factor > minimum_peak_points:
                    number_of_peaks += 1
                    b = int(begin // interpolation_factor)
                    e = int((n + 2) // interpolation_factor)
                    borders_roi.append([b, e])
                # re-init the begin and peak wide for next peak
                begin = -1
                peak_wide = 0
        # process the non-end peak
        if begin != -1 and peak_wide * interpolation_factor > minimum_peak_points:
            number_of_peaks += 1
            b = int(begin // interpolation_factor)
            e = int((n + 2) // interpolation_factor)
            borders_roi.append([b, e])

        return borders_roi


def infer_from_json_file(model: PeakDetector, file_path, save_path):
    # parse the json file
    raw_signals = []
    for file in os.listdir(file_path):
        if file.endswith('json'):
            with open(os.path.join(file_path, file)) as json_file:
                roi = json.load(json_file)
                raw_signal = roi['intensity']  # signal
                code = os.path.splitext(file)[0]  # signal title
                raw_index = roi['rt']  # signal index in the raw data
                raw_signals.append([code, raw_index, raw_signal])
    # infer and post process
    figure = plt.figure()
    for data in tqdm(raw_signals):
        signal_code, signal_index, signal_intensity = data
        label, borders, _ = model(np.array(signal_intensity))
        # plot
        figure.clear()
        ax = figure.add_subplot(111)
        # ax.plot(range(signal_index[0], signal_index[1]), signal_intensity, label=signal_code)
        ax.plot(signal_intensity, label=signal_code)
        title = "predicted border for {}".format(signal_code)
        if borders:
            for border in borders:
                begin, end = border
                ax.fill_between(range(begin, end + 1), y1=signal_intensity[begin:end + 1], y2=min(signal_intensity),
                                alpha=0.5)
        ax.set_title(title)
        ax.legend(loc='best')
        plt.savefig(save_path + '/{}.jpg'.format(signal_code))


def infer_from_txt_file(model: PeakDetector, file_path, save_path):

    full_signal, _ = read_from_txt(file_path)
    signalLen = len(full_signal)
    startPos = 0
    sliceLen = 200
    figure = plt.figure()
    pbar = tqdm(total=signalLen)
    while startPos <= signalLen - sliceLen:
        signalSlice = np.array(full_signal[startPos: startPos + sliceLen], dtype=np.float32)
        label, borders, _ = model(signalSlice)
        if label == 1:
            # plot
            figure.clear()
            signal_code = str(startPos)
            ax = figure.add_subplot(111)
            ax.plot(signalSlice, label=signal_code)
            title = "predicted border for {}".format(signal_code)
            if borders:
                for border in borders:
                    begin, end = border
                    ax.fill_between(range(begin, end + 1), y1=signalSlice[begin:end + 1], y2=min(signalSlice),
                                    alpha=0.5)
            ax.set_title(title)
            ax.legend(loc='best')
            plt.savefig(save_path + '/{}.jpg'.format(signal_code))
        startPos += sliceLen
        pbar.update(sliceLen)
    pbar.close()

if __name__ == "__main__":

    # parameters
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    classifierModelPath = "../data/weights/2022_0303/Classifier"
    segmentatorModelPath = "../data/weights/2022_0303/Segmentator"

    dataRoot = r"../data\raw_txt_data\MultiChannel_MDA_DEAD_20220305"
    # 文件名
    fileName = "ALL_2psi.txt"
    filePath = os.path.join(dataRoot, fileName)
    savePath = os.path.join(dataRoot, os.path.splitext(fileName)[0])
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # init the model
    peakDetector = PeakDetector(classifierModelPath, segmentatorModelPath, DEVICE)

    # infer from file
    # infer_from_json_file(peakDetector, filePath, savePath)
    infer_from_txt_file(peakDetector, filePath, savePath)
