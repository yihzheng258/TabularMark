import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

n = 150 # number of key cells
gamma = 1/2 # ratio between the length of green domain and red domain

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=10000)
# Parse arguments.

args = parser.parse_args()
seed = args.seed
dataset='HOG'


z_scores = []
for seed in range(10000, 10001):
    loaded_results = np.load(f"/home/zhengyihao/TabularMark/datasets/HOG/watermarked/{dataset}-{seed}.npy", allow_pickle=True).item()
    watermarked_data = loaded_results['watermarked_data']
    divide_seeds = loaded_results['divide_seeds']
    indices = loaded_results['indices'] 
    #添加水印
    digit_types = watermarked_data['target'].unique()
    digit_types.sort()
    green_cell = 0
    for idx, divide_seed in zip(indices, divide_seeds):
        np.random.seed(divide_seed)
        candidate_set = digit_types
        # 打乱cover_types的顺序
        shuffled_cover_types = list(digit_types)
        # print(shuffled_cover_types)
        np.random.shuffle(shuffled_cover_types)

        # 确保cover_types能被划分为两个相等大小的部分
        half_size = len(shuffled_cover_types) // 2

        # 划分成green_domain和red_domain
        green_domain = shuffled_cover_types[:half_size]
        red_domain = shuffled_cover_types[half_size:]

        if watermarked_data.loc[idx, 'target'] in green_domain:
            green_cell += 1
        
    z_score = (green_cell - n/2) / math.sqrt(n/4)
    z_scores.append(z_score)

print("The average z-score is ",np.mean(z_scores))





