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
import logging
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


def parsePeaks(peak: list, borders: list):
    """ 从peak中抽取pp值和travelTime的信息
        暂时只考虑一个峰值的情况

    Args:
        peak (list): peak signal slice
        borders (_type_): border of the peak
    Returns:
        
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
        return (travelTime, ppVal)
        ##### 以下为方便作图
        # if ppVal > 0.0012:
        #     return None
        # else:
        #     return (travelTime, ppVal)

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
        return (travelTime, ppVal)
        ###### 以下为方便作图
        # # if ppVal > 0.0012:
        # if ppVal > 0.0003 or travelTime > 60:
        #     return None
        # # else:
        # #     return (travelTime, ppVal)
    elif not borders:
        return None


def save_and_plot(
    save_folder:str, 
    raw_signal:list, 
    borders:list, 
    pred_prob:list, 
    timestamp:list, 
    count:int, 
    flag:str,
    plot_online=True
    ):
    # TODO 开启新线程来处理保存和绘图 以免影响消费者线程
    now = datetime.now()
    date = now.strftime("%Y_%m_%d")
    capture_time = now.strftime("%H_%M_%S_%f")
    signal_number = str(count).zfill(5)
    signal_code = "signal_" + signal_number + "_" + flag + "_" + capture_time

    # plot
    if plot_online:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.plot(raw_signal, label=signal_code)
        title = "signal NO.{}, capture time:{}, {}".format(signal_number, capture_time, flag)
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


def save_and_plot_doubleChannel(
    save_folder:str, 
    raw_signal1:list, 
    raw_signal2:list, 
    borders1:list, 
    borders2:list, 
    # pred_prob:list, 
    timestamp1:list, 
    timestamp2:list,
    count:int, 
    plot_online=True
    ):
    # TODO 开启新线程来处理保存和绘图 以免影响消费者线程
    now = datetime.now()
    date = now.strftime("%Y_%m_%d")
    capture_time = now.strftime("%H_%M_%S_%f")
    signal_number = str(count).zfill(5)
    signal_title = "signal_" + signal_number + "_" + capture_time
    signal_code1 = "signalChannel1_" + signal_number + "_" + capture_time
    signal_code2 = "signalChannel2_" + signal_number + "_" + capture_time

    # plot
    if plot_online:
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.plot(raw_signal1, label=signal_code1)
        title = "signal NO.{}, capture time:{}".format(signal_number, capture_time)
        if borders1:
            for border in borders1:
                begin, end = border
                ax.fill_between(range(begin, end + 1), y1=raw_signal1[begin:end + 1], y2=min(raw_signal1), alpha=0.5)
        ax.plot(raw_signal2, label=signal_code2)
        if borders2:
            for border in borders2:
                begin, end = border
                ax.fill_between(range(begin, end + 1), y1=raw_signal2[begin:end + 1], y2=min(raw_signal2), alpha=0.5)
        ax.set_title(title)
        ax.legend(loc='best')
        plt.savefig(save_folder + '/{}.jpg'.format(signal_title))
        plt.close("all")

    # save in json
    save_content = {
        "signal_number": signal_number,
        "signal_code": signal_code1,
        "raw_signal1": raw_signal1,
        "timestamp1": timestamp1,
        "borders1": borders1,
        "raw_signal2": raw_signal2,
        "timestamp2": timestamp2,
        "borders2": borders2,
        # "pred_prob": pred_prob,
        "capture_time": capture_time,
    }
    with open(save_folder + "/{}.json".format(signal_title), mode="w", encoding="utf-8") as f:
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


def plot_signal_borders(signal: list, borders: list, save_path, figure, title, label):
    figure.clear()
    ax = figure.add_subplot(111)
    ax.plot(signal, label=label)
    title = title
    if borders:
        for border in borders:
            begin, end = border
            ax.fill_between(range(begin, end + 1), y1=signal[begin:end + 1], y2=min(signal), alpha=0.5)
    ax.set_title(title)
    ax.legend(loc='best')
    plt.savefig(save_path)


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """绘制训练和验证集的loss曲线/acc曲线

    Args:
        train_x : epoch num range -> x axis of trian figure
        train_y : y axis of train figure
        valid_x : epoch num range -> x axis of valid figure
        valid_y : y axis of valid figure
        mode(str) : 'loss' or 'acc'
        out_dir : save path of the figure
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


def create_folder(exp_path):
    now = datetime.now()
    folder_name = now.strftime("%H_%M_%S")
    date = now.strftime("%Y_%m_%d")
    result_save_folder = os.path.join(exp_path, date, folder_name)
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)
    return result_save_folder


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def create_logger(log_root="./log"):
    log_dir = create_folder(log_root)
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


def check_data_dir(path_tmp):
    assert os.path.exists(path_tmp), \
        "\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath(path_tmp))


if __name__ == "__main__":
    exp_path = "./exp"
    create_folder(exp_path)
    create_logger()