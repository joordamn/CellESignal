import os, sys, shutil
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.chdir(sys.path[-1])

import numpy as np
import json
import matplotlib.pyplot as plt
import torch


def preprocess(x):
    x[x < 1e-5] = 0
    pos = (x != 0)

    x[pos] = torch.log10(x[pos])
    # x[pos] *= 10000
    x = torch.sigmoid(x)
    return x

root = r"../data\data_collection_20220115\txt_data\01"
noise_root = os.path.join(root, "noise_data")
peak_root = os.path.join(root, "peak_data")
noise_files = os.listdir(noise_root)
peak_files = os.listdir(peak_root)

for i in range(len(peak_files)):
    if noise_files[i].endswith("json"):
        noise_file = os.path.join(noise_root, noise_files[i])
        peak_file = os.path.join(peak_root, peak_files[i])

        with open(noise_file) as nf:
            noise = json.load(nf)
            noise_data = noise["intensity"]
        with open(peak_file) as pf:
            peak = json.load(pf)
            peak_data = peak["intensity"]



        noise_data = torch.tensor(noise_data)
        peak_data = torch.tensor(peak_data)

        noise_data = preprocess(noise_data)
        peak_data = preprocess(peak_data)

        plt.figure()
        plt.subplot(121)
        plt.plot(noise_data)
        plt.ylim(0,0.2)
        plt.subplot(122)
        plt.plot(peak_data)
        plt.ylim(0,0.2)
        plt.show()