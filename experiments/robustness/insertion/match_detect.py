# %%
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

n = 300 # number of key cells
gamma = 1/2 # ratio between the length of green domain and red domain

# parser = argparse.ArgumentParser()
# parser.add_argument("-seed", type=int, default=10000)
# # Parse arguments.

# args = parser.parse_args()
seed = 10000
dataset='covertype'
proportions = [0.2, 0.4, 0.6, 0.8, 1.0]

def match_tuples(origin_data, watermarked_data, indices):
    # 初始化匹配索引集合
    match_indices = []
    
    # 选择 MSBs 的 k 个属性作为主键
    primary_key_cols = ['Elevation', 'Aspect']  # 选择 'Elevation' 和 'Aspect' 作为主键

    # 创建一个字典来存储水印数据的主键值及其索引，确保只存储第一个出现的映射
    watermarked_dict = {}
    for new_idx, row in watermarked_data.iterrows():
        # print("new_idx: ",new_idx)
        key_dw = tuple(row[primary_key_cols])
        if key_dw not in watermarked_dict:
            watermarked_dict[key_dw] = new_idx

    # 对每个原始数据中的元组进行迭代
    for idx in indices:
        # 提取原始数据中的主键值
        key_do = tuple(origin_data.loc[idx, primary_key_cols])
        # 查找水印数据中的主键值
        if key_do in watermarked_dict:
            match_indices.append(watermarked_dict[key_do])
        else:
            # 如果没有匹配到，则插入 -1
            match_indices.append(-1)
    return match_indices


for proportion in proportions:
    z_scores = []
    for seed in range(10000, 10001):
        loaded_results = np.load(f"/home/zhengyihao/TabularMark/datasets/covertype/watermarked/{dataset}-{seed}.npy", allow_pickle=True).item()
        # watermarked_data = loaded_results['watermarked_data']
        watermarked_data = pd.read_csv(f"/home/zhengyihao/TabularMark/datasets/covertype/insertion/covertype-{seed}-{proportion}-0.csv")
        divide_seeds = loaded_results['divide_seeds']
        indices = loaded_results['indices']
        
        #添加水印
        cover_types = watermarked_data['Cover_Type'].unique()
        cover_types.sort()  
        green_cell = 0
        
        match_indices = match_tuples(loaded_results['watermarked_data'], watermarked_data, indices)
        
        num_match_indices = 0
        print("len match_indices: ",len(match_indices))
        for idx, divide_seed in zip(match_indices, divide_seeds):
            if idx == -1:
                continue
            num_match_indices += 1
            np.random.seed(divide_seed)
            candidate_set = cover_types
            # 打乱cover_types的顺序
            shuffled_cover_types = list(cover_types)
            # print(shuffled_cover_types)
            np.random.shuffle(shuffled_cover_types)

            # 确保cover_types能被划分为两个相等大小的部分
            half_size = len(shuffled_cover_types) // 2

            # 划分成green_domain和red_domain
            green_domain = shuffled_cover_types[:half_size]
            red_domain = shuffled_cover_types[half_size:]

            if watermarked_data.loc[idx, 'Cover_Type'] in green_domain:
                green_cell += 1
        
        print(num_match_indices) 
        z_score = (green_cell - n/2) / math.sqrt(n/4)
        z_scores.append(z_score)

    print(f"The average z-score of {proportion} is ",np.mean(z_scores))






