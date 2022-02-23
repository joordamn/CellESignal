# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   utils.py
@Time    :   2022/01/07 19:14:00
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   utils scripts
-------------------------
'''

import time, os, json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def read_from_txt(file_path):
    """read data from txt file
    signal content:
            % ...
            % ...
            float; float
            ...
    border content:
            int \t int
            int \t int
            ...

    Args:
        file_path (str): txt file path

    Returns:
        data (list): signal data(float) or border data(int)
    """

    with open(file_path, mode='r', encoding='utf-8') as f:
        flag = ""
        data = []

        lines = f.readlines()
        if lines[0].startswith("%"):
            del lines[0:5]
        if ";" in lines[5]:
            # 此时返回信号数据
            flag = "signal"
            for line in lines:
                tar = line.split(";")[1]
                data.append(float(tar))
        else:
            # 此时返回border数据
            flag = "border"
            for line in lines:
                begin, end = line.split("\t")
                data.append([int(begin), int(end)])

    return data, flag


def save_and_plot(save_folder:str, raw_signal:list, borders:list, pred_prob:list, timestamp:list, count:int, plot_online=True):
    # TODO 开启新线程来处理保存和绘图 以免影响消费者线程
    now = datetime.now()
    date = now.strftime("%Y_%m_%d")
    capture_time = now.strftime("%H_%M_%S_%f")
    signal_number = str(count).zfill(5)
    signal_code = "signal_" + signal_number + "_" + capture_time

    # plot
    if plot_online:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.plot(raw_signal, label=signal_code)
        title = "signal NO.{}, capture time:{}".format(signal_number, capture_time)
        if borders:
            for border in borders:
                begin, end = border
                ax.fill_between(range(begin, end + 1), y1=raw_signal[begin:end + 1], y2=min(raw_signal), alpha=0.5)
        ax.set_title(title)
        ax.legend(loc='best')
        plt.savefig(save_folder + '/{}.jpg'.format(signal_code))
        plt.close("all")

    # save in json
    save_content = {
        "signal_number": signal_number,
        "signal_code": signal_code,
        "raw_signal": raw_signal,
        "timestamp": timestamp,
        "borders": borders,
        "pred_prob": pred_prob,
        "capture_time": capture_time,
    }
    with open(save_folder + "/{}.json".format(signal_code), mode="w", encoding="utf-8") as f:
        json.dump(save_content, f, indent=4)


def post_plot(json_folder):
    print("start plotting")
    figure = plt.figure()
    for file in tqdm(os.listdir(json_folder)):
        if file.endswith('json'):
            with open(os.path.join(json_folder, file), 'r') as json_file:
                content = json.load(json_file)
                raw_signal = content["raw_signal"]
                signal_code = content["signal_code"]
                signal_number = content["signal_number"]
                borders = content["borders"]
                capture_time = content["capture_time"]
                
                figure.clear()
                ax = figure.add_subplot(111)
                ax.plot(raw_signal, label=signal_code)
                title = "signal NO.{}, capture time:{}".format(signal_number, capture_time)
                if borders:
                    for border in borders:
                        begin, end = border
                        ax.fill_between(range(begin, end + 1), y1=raw_signal[begin:end + 1], y2=min(raw_signal), alpha=0.5)
                ax.set_title(title)
                ax.legend(loc='best')
                plt.savefig(json_folder + '/{}.jpg'.format(signal_code))


def create_folder(exp_path):
    now = datetime.now()
    folder_name = now.strftime("%H_%M_%S")
    date = now.strftime("%Y_%m_%d")
    result_save_folder = os.path.join(exp_path, date, folder_name)
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)
    return result_save_folder

if __name__ == "__main__":
    exp_path = "./exp"
    create_folder(exp_path)