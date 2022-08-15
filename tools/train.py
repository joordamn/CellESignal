# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   train.py
@Time    :   2022/03/02 01:24:53
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   
-------------------------
'''


import os, sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.training_config import cfg
from tools.trainer import CombinedLoss, Trainer, accuracy, iou
from tools.scheduler import CosineWarmupLr
from dataset.dataset import ROIDataset
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator
from utils.utils import create_logger, check_data_dir, plot_line


parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--type', default="segment", help='train model type', type=str)
parser.add_argument('--lr', default=None, help='learning rate', type=float)
parser.add_argument('--max_epoch', default=None, type=int)
parser.add_argument('--train_bs', default=0, type=int)
parser.add_argument('--data_root_dir', default=r"../data/train_data/2022_03_03/segment/",
                    help="path to your dataset")
args = parser.parse_args()

cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.train_bs if args.train_bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch
cfg.model_type = args.type if args.type else cfg.model_type


if __name__ == "__main__":

    # logger init
    logger, log_dir = create_logger()

    # ----------------step 1/4: loading data----------------------#
    root_dir = args.data_root_dir
    train_dir = os.path.join(root_dir, "train")
    valid_dir = os.path.join(root_dir, "val")
    check_data_dir(train_dir)
    check_data_dir(valid_dir)

    train_set = ROIDataset(train_dir, cfg.device, interpolate=True, length=cfg.length, model_type=cfg.model_type)
    valid_set = ROIDataset(valid_dir, cfg.device, interpolate=True, length=cfg.length, model_type=cfg.model_type)

    train_loader = DataLoader(train_set, batch_size=cfg.train_bs, shuffle=True)#, num_workers=cfg.workers)
    valid_loader = DataLoader(valid_set, batch_size=cfg.valid_bs)#, num_workers=cfg.workers)

    # ----------------step 2/4: net definition and loss functin--------------------#
    if cfg.model_type == "classification":
        model = Classifier().to(cfg.device)
        loss_f = nn.CrossEntropyLoss()
        acc_metric = accuracy
    elif cfg.model_type == "segment":
        model = Segmentator().to(cfg.device)
        # todo replace with nn.BCEwith...
        loss_f = CombinedLoss([0.4, 0.2])
        acc_metric = iou
    else:
        raise TypeError("model type wrong, please check cfg.model_type")

    # -----------step 3/4: optimizer and scheduler -----------#
    # optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr_init)

    if cfg.is_warmup:
        # 注意，如果有这段warmup代码，一定要到trainer中修改 scheduler.step()
        iter_per_epoch = len(train_loader)
        scheduler_warm = CosineWarmupLr(optimizer, batches=iter_per_epoch, max_epochs=cfg.max_epoch,
                                   base_lr=cfg.lr_init, final_lr=cfg.lr_final,
                                   warmup_epochs=cfg.warmup_epochs, warmup_init_lr=cfg.lr_warmup_init)
    else:
        scheduler_warm = None
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # ----------step 4/4: training ------------------------#
    logger.info(
        "train model type: {}\n"
        "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n"
        "device:{}"
        .format(cfg.model_type, cfg, loss_f, scheduler, optimizer, cfg.device)#torch.cuda.get_device_name())
        )

    loss_rec = {"train": [], "val": []}  # loss 记录
    acc_rec = {"train": [], "val": []}  # acc 记录
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(cfg.max_epoch):
        # trian
        loss_train, acc_train = Trainer.train(
            data_loader=train_loader,
            model=model,
            loss_f=loss_f,
            optimizer=optimizer,
            cfg=cfg,
            epoch_idx=epoch,
            logger=logger,
            acc_metric=acc_metric,
            scheduler=scheduler_warm
        )
        # val
        loss_val, acc_val = Trainer.valid(
            data_loader=valid_loader,
            model=model,
            loss_f=loss_f,
            acc_metric=acc_metric,
            cfg=cfg,
        )

        # lr update
        if not cfg.is_warmup:
            scheduler.step()
        
        logger.info(
            "Epoch[{:0>3}/{:0>3}] \t Train loss: {:.6f} \t Train Acc: {:.4f} \t Valid Acc:{:.4f} \t LR:{} \n"
            .format(epoch, cfg.max_epoch, loss_train, acc_train, acc_val, optimizer.param_groups[0]["lr"])
        )

        # record train info
        loss_rec["train"].append(loss_train), loss_rec["val"].append(loss_val)
        acc_rec["train"].append(acc_train), acc_rec["val"].append(acc_val)

        # visualization
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["val"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["val"], mode="acc", out_dir=log_dir)

        # save model
        if best_acc < acc_val or epoch == cfg.max_epoch-1:
            if best_acc < acc_val:
                model_name = model.__class__.__name__
                best_epoch = epoch
                best_acc = acc_val
            else:
                model_name = model.__class__.__name__ + "_last"
                best_epoch = best_epoch
                best_acc = best_acc
            
            save_path = os.path.join(log_dir, model_name)
            torch.save(model.state_dict(), save_path)
            logger.info(
                "Best in Epoch {}, acc: {:.4f}".format(best_epoch, best_acc)
            )
    
    # finish
    logger.info(
        "{} trianing done, best acc: {:.4f}, in Epoch {}"
        .format(cfg.model_type, best_acc, best_epoch)
    )