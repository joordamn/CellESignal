# -*- encoding: utf-8 -*-
'''
-------------------------
@File    :   trainer.py
@Time    :   2022/03/01 21:26:41
@Author  :   Zhongning Jiang 
@Contact :   zhonjiang8-c@my.cityu.edu.hk
@Desc    :   训练函数及loss function
-------------------------
'''


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def accuracy(logits, y_true):
    """
    :param logits: np.ndarray, output of the model
    :param y_true: np.ndarray
    """
    predictions = np.argmax(logits, axis=1)
    correct_samples = np.sum(predictions == y_true)
    total_samples = y_true.shape[0]
    return float(correct_samples) / total_samples


def iou(logits, y_true, smooth=1e-2):
    """
    :param logits: np.ndarray, output of the model
    :param y_true: np.ndarray
    :param smooth: float
    """
    batch_size, channels, samples = logits.shape
    values = np.zeros(channels)

    for i in range(channels):
        pred = logits[:, i, :] > 0.5
        gt = y_true[:, :].astype(np.bool)
        intersection = (pred & gt).sum(axis=1)
        union = (pred | gt).sum(axis=1)
        values[i] = np.mean((intersection + smooth) / (union + smooth))

    return np.mean(values)


class WeightedBCE(nn.Module):
    def __init__(self, weights=None):
        super(WeightedBCE, self).__init__()
        self.weights = weights
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (target * self.logsigmoid(output)) + \
                self.weights[0] * ((1 - target) * self.logsigmoid(-output))
        else:
            loss = target * self.logsigmoid(output) + (1 - target) * self.logsigmoid(-output)
        return torch.neg(torch.mean(loss))


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        # predict = predict.sigmoid()
        # numerator = torch.sum(predict * target, dim=1)
        # denominator = torch.sum(torch.sqrt(predict) + target, dim=1)
        # score = 1 - torch.mean((2 * numerator + self.epsilon) / (denominator + self.epsilon))

        num = predict.size(0)
        # pred不需要转bool变量，如https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py#L44
        # soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
        pred = torch.sigmoid(predict).view(num, -1)
        targ = target.view(num, -1)

        intersection = (pred * targ).sum()  # 利用预测值与标签相乘当作交集
        union = (pred + targ).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = WeightedBCE(weights)

    def forward(self, output, target):
        return self.dice(output, target) + self.bce(output, target)


class Trainer:
    
    @staticmethod
    def train(
        data_loader: DataLoader, 
        model: nn.Module, 
        loss_f,
        optimizer, 
        cfg, 
        epoch_idx, 
        logger,
        acc_metric,
        scheduler=None,
        ):

        model.train()
        loss_sigma = []
        acc_accum = 0
        sample_count = 0 # total sample count of each epoch
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            # forward
            outputs = model(inputs)
            # backward
            
            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # acc and loss cal
            acc_accum += acc_metric(
                            outputs.detach().cpu().numpy(), 
                            labels.detach().cpu().numpy()
                            ) * len(inputs)
            loss_sigma.append(loss.item())
            sample_count += len(inputs)

            # print train info by interval
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info('|Epoch[{}/{}]||batch[{}/{}]||batch_loss: {:.6f}||accuracy: {:.4f}|'.format(
                    epoch_idx, cfg.max_epoch, i + 1, len(data_loader), loss.item(), float(acc_accum / sample_count)))
        
        # cal mean acc and loss
        loss_mean = np.mean(loss_sigma) # mean loss of each epoch
        acc_mean = np.mean(float(acc_accum / sample_count)) # mean acc of each epoch
        
        if scheduler:
            scheduler.step()
        
        return loss_mean, acc_mean
            
        
    @staticmethod
    def valid(
        data_loader: DataLoader, 
        model: nn.Module, 
        loss_f, 
        cfg,
        acc_metric,
        ):

        model.eval()
        loss_sigma = []
        val_acc_accum = 0
        sample_count = 0 # total sample count of valid epoch

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            # forward
            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())
            # accum loss
            loss_sigma.append(loss.item())
            # evaluate
            val_acc_accum += acc_metric(
                                outputs.detach().cpu().numpy(), 
                                labels.detach().cpu().numpy()
                                ) * len(inputs)
            sample_count += len(inputs)
        
        loss_mean = np.mean(loss_sigma)
        val_acc_mean = np.mean(float(val_acc_accum / sample_count))
        return loss_mean, val_acc_mean