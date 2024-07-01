import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 150 # number of key cells
gamma = 1/2 # ratio between the length of green domain and red domain

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=10000)
# Parse arguments.

args = parser.parse_args()
seed = args.seed
dataset='HOG'

original_file = '../../datasets/HOG/digits_HOG.csv'
origin = pd.read_csv(original_file)

np.random.seed(seed)

divide_seeds = np.random.randint(0, 2**32 - 1, size=n)

if len(origin) < n:
    raise ValueError("data中的记录数小于所请求的n个记录")

indices = np.random.choice(len(origin), size=n, replace=False)

#添加水印
digit_types = origin['target'].unique()
# 验证索引列表和种子列表的长度是否一致
if len(indices) != len(divide_seeds):
    raise ValueError("索引文件和种子文件的长度不一致")
digit_types.sort()

# 开始替换Cover_Type值
for idx, divide_seed in zip(indices, divide_seeds):
    np.random.seed(divide_seed)
    candidate_set = digit_types
     # 打乱cover_types的顺序
    shuffled_cover_types = list(digit_types)

    np.random.shuffle(shuffled_cover_types)

    # 确保cover_types能被划分为两个相等大小的部分
    half_size = len(shuffled_cover_types) // 2

    # 划分成green_domain和red_domain
    green_domain = shuffled_cover_types[:half_size]
    red_domain = shuffled_cover_types[half_size:]

    perturb_value = np.random.choice(green_domain)

    # 将df中对应索引的Cover_Type属性替换为这个选定的值
    origin.loc[idx, 'target'] = perturb_value
    
results = {
    'watermarked_data': origin,
    'divide_seeds': divide_seeds,
    'indices': indices
}

np.save(f"../../datasets/HOG/watermarked/{dataset}-{seed}.npy", results)




