import pandas as pd
from sklearn.decomposition import PCA
from app import path
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import csv

num = 5   # 移動平均　無しの場合は値を1に
b = np.ones(num)/num

N = int((path.data_long / 8) - 1)
dt = 0.002
fc = 20
# hm = 60
# fq = np.linspace(0, 1.0 / dt, 126)
train_x = []
train_y = []

# 外れ値の関数
def outlier_iqr(df):

    for i in range(len(df.columns)):
        # 四分位数
        a = df.iloc[:, i]

        Q1 = a.quantile(.25)
        Q3 = a.quantile(.75)
        # 四分位数
        q = Q3 - Q1

        # 外れ値の基準点
        outlier_min = Q1 - q * 3
        outlier_max = Q3 + q * 3

        # 範囲から外れている値を除く
        a[a < outlier_min] = outlier_min
        a[a > outlier_max] = outlier_max

    return df

# ファイルの読み込み
for i in path.new_subject:
    for d in path.new_day:
        for s in path.sets:
            for j in path.mix_char2:
                y1 = 0
                y2 = 0
                y3 = 0
                y4 = 0
                y5 = 0
                y6 = 0
                y7 = 0
                y8 = 0
                for l in path.time:
                    y1_0 = 0
                    y2_0 = 0
                    y3_0 = 0
                    y4_0 = 0
                    y5_0 = 0
                    y6_0 = 0
                    y7_0 = 0
                    y8_0 = 0
                    for o in path.cut8_time:
                        file_path = path.cut_8_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        df = pd.read_csv(file_path)
                        df = outlier_iqr(df)
                        # f_df = np.fft.fft(df)
                        # print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                        # print(df)

                        ch0 = df.iloc[:, 0].values
                        ch1 = df.iloc[:, 1].values
                        ch2 = df.iloc[:, 2].values
                        ch3 = df.iloc[:, 3].values
                        ch4 = df.iloc[:, 4].values
                        ch5 = df.iloc[:, 5].values
                        ch6 = df.iloc[:, 6].values
                        ch7 = df.iloc[:, 7].values