import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

mu = 0
sigma = 20

n = 300 # number of key cells
gamma = 1/2 # ratio between the length of green domain and red domain
p = 2 * sigma
k = 500

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=10000)
# Parse arguments.

args = parser.parse_args()
seed = args.seed
dataset='synthetic'

original_file = '../../datasets/synthetic_dataset/synthetic_data.csv'
origin = pd.read_csv(original_file)


z_scores = []
for seed in range(10000, 10001):
    loaded_results = np.load(f"../../datasets/synthetic_dataset/watermarked/{dataset}-{seed}.npy", allow_pickle=True).item()
    # watermarked_data = loaded_results['watermarked_data']
    watermarked_data = pd.read_csv('../../datasets/synthetic_dataset/ROC/synthetic-10000-0.6-50.csv')
    divide_seeds = loaded_results['divide_seeds']
    indices = loaded_results['indices']
    green_cell = 0
    for idx, divide_seed in zip(indices, divide_seeds):
        np.random.seed(divide_seed)
        # 生成等分点
        intervals = np.linspace(-p, p, k + 1)
        # 将 [-p, p] 等分为 k 份
        segments = [(intervals[i], intervals[i + 1]) for i in range(k)]
        np.random.shuffle(segments)
        # 将 segments 分为 green domains 和 red domains
        half_k = k // 2
        green_domains = segments[:half_k]
        red_domains = segments[half_k:]

        difference = watermarked_data.loc[idx, 'dimension_0'] - origin.loc[idx, 'dimension_0']
        for low, high in green_domains:
            if low <= difference < high:
                green_cell += 1
                break
        
    z_score = (green_cell - n/2) / math.sqrt(n/4)
    z_scores.append(z_score)

print("The average z-score is ",np.mean(z_scores))





