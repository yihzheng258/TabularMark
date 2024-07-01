# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import numpy as np

dataset = 'covertype'
seed = 10000

# 读取数据
origin = pd.read_csv("../../../datasets/covertype/cover_type_with_columns.csv")

proportions = [0.2, 0.4, 0.6, 0.8, 1.0]

for proportion in proportions:
    loaded_results = np.load(f"../../../datasets/covertype/watermarked/{dataset}-{seed}.npy", allow_pickle=True).item()
    # watermarked_data = loaded_results['watermarked_data']
    watermarked_data = pd.read_csv(f"../../../datasets/covertype/alteration/{dataset}-{seed}-{proportion}-{0}.csv")
    divide_seeds = loaded_results['divide_seeds']
    indices = loaded_results['indices']


    # 分离特征和目标变量
    X = watermarked_data.drop(columns=['Cover_Type'])
    y = watermarked_data['Cover_Type']

    # 将目标变量进行标签编码
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 划分训练集和测试集
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    # 分离特征和目标变量
    X = origin.drop(columns=['Cover_Type'])
    y = origin['Cover_Type']

    # 将目标变量进行标签编码
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 划分训练集和测试集
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 定义XGBoost模型
    model = xgb.XGBClassifier(n_estimators=30, max_depth=10, n_jobs=4)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算每个类别的F1-scores
    f1_scores = f1_score(y_test, y_pred, average=None)

    # 输出F1-scores
    for i, score in enumerate(f1_scores):
        print(f"{proportion}: Category {le.inverse_transform([i])[0]}: F1-score = {score:.4f}")



