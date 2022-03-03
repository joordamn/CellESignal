# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   config.py
@Time    :   2022/03/01 21:46:08
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   config for training process
-------------------------
'''

import torch
from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

# cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.device = "cpu"
cfg.max_epoch = 200
cfg.log_interval = 10

# 信号长度
cfg.length = 256

# 模型类别
cfg.model_type = "classification" # "classification" or "segment"

# batch size
cfg.train_bs = 256  # 8
cfg.valid_bs = 8    # 4
cfg.workers = 1  # 16

# 学习率
cfg.lr_init = 1e-3  # pretraied_model::0.1
cfg.factor = 0.1
cfg.milestones = [75, 130]
cfg.weight_decay = 5e-4
cfg.momentum = 0.9

# warmup cosine decay
cfg.is_warmup = False
cfg.warmup_epochs = 1
cfg.lr_final = 1e-6
cfg.lr_warmup_init = 0.  # 是0. 没错

cfg.hist_grad = False
