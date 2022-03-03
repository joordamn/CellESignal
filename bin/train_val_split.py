# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   train_val_split.py
@Time    :   2022/01/25 20:39:42
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   从data_collection中读取peak和noise的json文件，以划分训练和验证集
-------------------------
'''

from datetime import datetime
import os, sys, shutil
import random

from tqdm import tqdm
# sys.path.append("..")
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# os.chdir(sys.path[-1])


def get_classs_json_path(root_folder):
    json_path_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith("json"):
                json_path = os.path.join(root, file)
                json_path_list.append(json_path)
    return json_path_list


def get_segment_json_path(root_folder):
    json_path_list = []
    for root, dirs, files in os.walk(root_folder):
        if os.path.split(root)[-1] == "peak_data": # todo use re
            for file in files:
                if file.endswith("json"):
                    json_path = os.path.join(root, file)
                    json_path_list.append(json_path)
    return json_path_list


def train_split(json_path_list, split_ratio, train_folder, val_folder):
    # random split
    random.shuffle(json_path_list)

    # copy and rename to target folder
    for i, json_path in tqdm(enumerate(json_path_list)):
        new_name = "sample_" + str(i).zfill(4) + ".json"
        if i <= len(json_path_list) * split_ratio:
            dst_path = os.path.join(train_folder, new_name)
            shutil.copy(json_path, dst_path)
        else:
            dst_path = os.path.join(val_folder, new_name)
            shutil.copy(json_path, dst_path)


if __name__ == "__main__":
    SPLIT_RATIO = 0.8
    root_folders = [
        "../data/data_collection_20220115/txt_data/01",
        "../data/data_collection_20220115/txt_data/02",
        "../data/data_collection_20220115/txt_data/03",
        "../data/data_collection_20220301/txt_data/01",
        "../data/data_collection_20220301/txt_data/02",
    ]
    train_root = "../data/train_data/"
    date = datetime.now().strftime("%Y_%m_%d")
    
    # 数据集生成保存的路径
    class_train_folder = train_root + date + "/classification/train/"
    class_val_folder = train_root + date + "/classification/val/"
    segment_train_folder = train_root + date + "/segment/train/"
    segment_val_folder = train_root + date + "/segment/val/"

    try:
            shutil.rmtree(class_train_folder)
            shutil.rmtree(class_val_folder)
            shutil.rmtree(segment_train_folder)
            shutil.rmtree(segment_val_folder)
    except:
        pass
        
    if not os.path.exists(class_train_folder):
        os.makedirs(class_train_folder)
        os.makedirs(class_val_folder)
        os.makedirs(segment_train_folder)
        os.makedirs(segment_val_folder)
    
    # 分类数据集划分
    class_json_list = []
    for root_folder in root_folders:
        class_json_list.extend(get_classs_json_path(root_folder))
    
    train_split(
        json_path_list=class_json_list,
        split_ratio=SPLIT_RATIO,
        train_folder=class_train_folder,
        val_folder=class_val_folder,
    )

    # 分割数据集划分
    segment_json_list = []
    for root_folder in root_folders:
        segment_json_list.extend(get_segment_json_path(root_folder))

    train_split(
        json_path_list=segment_json_list,
        split_ratio=SPLIT_RATIO,
        train_folder=segment_train_folder,
        val_folder=segment_val_folder,
    )


    

