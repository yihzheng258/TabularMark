import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

n = 300 # number of key cells
gamma = 1/2 # ratio between the length of green domain and red domain

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=10000)
# Parse arguments.

args = parser.parse_args()
seed = args.seed
dataset='covertype'

z_scores = []
for seed in range(10000, 10001):
    loaded_results = np.load(f"/home/zhengyihao/TabularMark/datasets/covertype/watermarked/{dataset}-{seed}.npy", allow_pickle=True).item()
    # watermarked_data = loaded_results['watermarked_data']
    watermarked_data = pd.read_csv(f"/home/zhengyihao/TabularMark/datasets/covertype/alteration/covertype-{seed}-1.0-0.csv")
    divide_seeds = loaded_results['divide_seeds']
    indices = loaded_results['indices']
    #添加水印
    cover_types = watermarked_data['Cover_Type'].unique()
    cover_types.sort()  

    green_cell = 0
    for idx, divide_seed in zip(indices, divide_seeds):
        np.random.seed(divide_seed)
        candidate_set = cover_types
        # 打乱cover_types的顺序
        shuffled_cover_types = list(cover_types)
        # print(shuffled_cover_types)
        np.random.shuffle(shuffled_cover_types)

        half_size = len(shuffled_cover_types) // 2

        green_domain = shuffled_cover_types[:half_size]
        red_domain = shuffled_cover_types[half_size:]

        if watermarked_data.loc[idx, 'Cover_Type'] in green_domain:
            green_cell += 1
        
    z_score = (green_cell - n/2) / math.sqrt(n/4)
    z_scores.append(z_score)

print("The average z-score is ",np.mean(z_scores))





