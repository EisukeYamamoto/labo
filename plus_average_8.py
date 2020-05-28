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

N = (path.data_long / 8) - 1
dt = 0.001
fc = 20
# hm = 60
train_x = []
train_y = []

# 外れ値の関数
def outlier_iqr(df):

    for i in range(len(df.columns)):
        # 四分位数
        a = df.iloc[:, i]

        Q1 = a.quantile(.10)
        Q3 = a.quantile(.90)
        # 四分位数
        q = Q3 - Q1

        # 外れ値の基準点
        outlier_min = Q1 - q * 1.5
        outlier_max = Q3 + q * 1.5

        # 範囲から外れている値を除く
        a[a < outlier_min] = outlier_min
        a[a > outlier_max] = outlier_max

    return df

# ファイルの読み込み
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.mix_char2:
            y1 = 0
            y2 = 0
            y3 = 0
            y4 = 0
            y5 = 0
            y6 = 0
            y7 = 0
            for l in path.time:
                y1_0 = 0
                y2_0 = 0
                y3_0 = 0
                y4_0 = 0
                y5_0 = 0
                y6_0 = 0
                y7_0 = 0
                for o in path.cut8_time:
                    file_path = path.ifft_mix_8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "ifft.CSV"
                    df = pd.read_csv(file_path)
                    df = outlier_iqr(df)
                    f_df = np.fft.fft(df)
                    # print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                    # print(df)

                    ch0 = df.iloc[:, 0].values
                    ch1 = df.iloc[:, 1].values
                    ch2 = df.iloc[:, 2].values
                    ch3 = df.iloc[:, 3].values
                    ch4 = df.iloc[:, 4].values
                    ch5 = df.iloc[:, 5].values
                    ch6 = df.iloc[:, 6].values
                    # ch7 = list(df.iloc[:, 7].values)


                    ch0 = np.convolve(ch0, b, mode='same')
                    ch1 = np.convolve(ch1, b, mode='same')
                    ch2 = np.convolve(ch2, b, mode='same')
                    ch3 = np.convolve(ch3, b, mode='same')
                    ch4 = np.convolve(ch4, b, mode='same')
                    ch5 = np.convolve(ch5, b, mode='same')
                    ch6 = np.convolve(ch6, b, mode='same')

                    plt.plot(ch0)


                    y1_0 += ch0
                    y2_0 += ch1
                    y3_0 += ch2
                    y4_0 += ch3
                    y5_0 += ch4
                    y6_0 += ch5
                    y7_0 += ch6

                y1_0 = pd.DataFrame(y1_0 / 15)
                y2_0 = pd.DataFrame(y2_0 / 15)
                y3_0 = pd.DataFrame(y3_0 / 15)
                y4_0 = pd.DataFrame(y4_0 / 15)
                y5_0 = pd.DataFrame(y5_0 / 15)
                y6_0 = pd.DataFrame(y6_0 / 15)
                y7_0 = pd.DataFrame(y7_0 / 15)

                y1 += y1_0
                y2 += y2_0
                y3 += y3_0
                y4 += y4_0
                y5 += y5_0
                y6 += y6_0
                y7 += y7_0

            y1 = pd.DataFrame(y1 / 10)
            y2 = pd.DataFrame(y2 / 10)
            y3 = pd.DataFrame(y3 / 10)
            y4 = pd.DataFrame(y4 / 10)
            y5 = pd.DataFrame(y5 / 10)
            y6 = pd.DataFrame(y6 / 10)
            y7 = pd.DataFrame(y7 / 10)

            c0 = y1.values
            plt.plot(c0, c='r', linewidth=4)
            plt.show()


            nf = open(path.plus_avarage + "/" + d + "/" + i + "/" + j  + "/" + l + "ifft.CSV", 'w')
            dataWriter = csv.writer(nf)


            # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7, y8], axis=1)
            new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7], axis=1)

            # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
            new_data.columns = new_data.iloc[0, :]
            new_data.index = new_data.iloc[:, 0]
            # new_data = new_data.iloc[1:129, 1:8]
            new_data = new_data.iloc[1:129, 1:7]

            new_data.to_csv(path.plus_avarage + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV")
            print("fft/" + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "fft.CSV")

            nf.close()


