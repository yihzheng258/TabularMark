import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

np.random.seed(seed)

divide_seeds = np.random.randint(0, 2**32 - 1, size=n)

if len(origin) < n:
    raise ValueError("data中的记录数小于所请求的n个记录")

indices = np.random.choice(len(origin), size=n, replace=False)

# 验证索引列表和种子列表的长度是否一致
if len(indices) != len(divide_seeds):
    raise ValueError("索引文件和种子文件的长度不一致")

# 开始替换Cover_Type值
for idx, divide_seed in zip(indices, divide_seeds):
    temp = origin.copy()
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

    green_domain_values = [np.random.uniform(low, np.nextafter(high, low)) for low, high in green_domains]
    perturb_value = np.random.choice(green_domain_values)

    temp.loc[idx, 'dimension_0'] += perturb_value
    
results = {
    'watermarked_data': temp,
    'divide_seeds': divide_seeds,
    'indices': indices
}

np.save(f"../../datasets/synthetic_dataset/watermarked/{dataset}-{seed}.npy", results)




