import torch
import torch.nn as nn


def preprocessing(batch):
    batch_size, _, n_peaks = batch.shape
    processed_batch = batch.clone().view(batch_size, n_peaks)
    # TO DO: rewrite without loop
    for x in processed_batch:
        x[x < 1e-4] = 0
        pos = (x != 0)
        x[pos] = torch.log10(x[pos])
        x[pos] = x[pos] - torch.min(x[pos])
        x[pos] = x[pos] / torch.max(x[pos])
    return processed_batch.view(batch_size, 1, n_peaks)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=2, dilation=1, stride=1):
        super().__init__()

        self.basic_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        return self.basic_block(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock = nn.Sequential(
            Block(1, 8),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512)
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = preprocessing(x)
        x = self.convBlock(x)
        x, _ = torch.max(x, dim=2)
        return self.classification(x), None
