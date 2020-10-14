import pandas as pd
from sklearn.decomposition import PCA
from app import path
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import csv

hammingWindow = np.hamming(255)

num = 10  # 移動平均　無しの場合は値を1に
b = np.ones(num) / num

N = int((path.data_long / 8) - 1)
dt = 0.001
fc = 20
# hm = 60
# fq = np.linspace(0, 1.0 / dt, 126)
train_x = [ ]
train_y = [ ]


# 外れ値の関数
def outlier_iqr(df):
    for i in range(len(df.columns)):
        # 四分位数
        a = df.iloc[ :, i ]

        Q1 = a.quantile(.25)
        Q3 = a.quantile(.75)
        # 四分位数
        q = Q3 - Q1

        # 外れ値の基準点
        outlier_min = Q1 - q * 3
        outlier_max = Q3 + q * 3

        # 範囲から外れている値を除く
        a[ a < outlier_min ] = outlier_min
        a[ a > outlier_max ] = outlier_max

    return df


y1 = 0

# ファイルの読み込み
for i in path.new_subject:
    for d in path.new_day:
        for s in path.sets:
            for j in path.mix_char2:
                for l in path.time:
                    for o in path.cut8_time:
                        file_path = path.cut_8_new + "/" + i + "/" + d + "/" + s + "/" \
                                    + j + "/" + o + "/" + l + "cut.CSV"
                        print(
                            "////////////////////////////////////////////////////////////////////////////////" +
                            "//////////////////////")
                        print(
                            "file : " + path.cut_8_new + "/" + i + "/" + d + "/" + s + "/"
                            + j + "/" + o + "/" + l + "cut.CSV")

                        df = pd.read_csv(file_path)
                        df = outlier_iqr(df)
                        # f_df = np.fft.fft(df)
                        # print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                        # print(df)

                        y1 = df.iloc[ :, 0 ].values
                        y2 = df.iloc[ :, 1 ].values
                        y3 = df.iloc[ :, 2 ].values
                        y4 = df.iloc[ :, 3 ].values
                        y5 = df.iloc[ :, 4 ].values
                        y6 = df.iloc[ :, 5 ].values
                        y7 = df.iloc[ :, 6 ].values
                        y8 = df.iloc[ :, 7 ].values

                        # print(y1.shape)
                        x = np.linspace(0, 1 / dt, len(y1))
                        plt.plot(x, y1, c="r", alpha=0.7, label='filtered')
                        plt.savefig(path.preprocessing_abs + "/" + i + "/" + d + "/" + s
                                    + "/" + o + "/" + j + "/plt1_init.png")
                        plt.close()
                        # plt.show()


                        y1 = hammingWindow * y1.T
                        y2 = hammingWindow * y2.T
                        y3 = hammingWindow * y3.T
                        y4 = hammingWindow * y4.T
                        y5 = hammingWindow * y5.T
                        y6 = hammingWindow * y6.T
                        y7 = hammingWindow * y7.T
                        y8 = hammingWindow * y8.T

                        ch0 = list(y1.T)
                        ch1 = list(y2.T)
                        ch2 = list(y3.T)
                        ch3 = list(y4.T)
                        ch4 = list(y5.T)
                        ch5 = list(y6.T)
                        ch6 = list(y7.T)
                        ch7 = list(y8.T)

                        n_ch0 = np.array(ch0)
                        n_ch1 = np.array(ch1)
                        n_ch2 = np.array(ch2)
                        n_ch3 = np.array(ch3)
                        n_ch4 = np.array(ch4)
                        n_ch5 = np.array(ch5)
                        n_ch6 = np.array(ch6)
                        n_ch7 = np.array(ch7)

                        f_ch0 = np.fft.fft(n_ch0)
                        f_ch1 = np.fft.fft(n_ch1)
                        f_ch2 = np.fft.fft(n_ch2)
                        f_ch3 = np.fft.fft(n_ch3)
                        f_ch4 = np.fft.fft(n_ch4)
                        f_ch5 = np.fft.fft(n_ch5)
                        f_ch6 = np.fft.fft(n_ch6)
                        f_ch7 = np.fft.fft(n_ch7)

                        x = np.linspace(0, 1 / dt, len(f_ch0))
                        plt.plot(x, f_ch0, c="r", alpha=0.7, label='filtered')
                        plt.savefig(path.preprocessing_abs + "/" + i + "/" + d + "/" + s
                                    + "/" + o + "/" + j + "/plt2_after_fft.png")
                        plt.close()
                        # plt.show()

                        c_f_ch0 = np.copy(f_ch0)
                        c_f_ch1 = np.copy(f_ch1)
                        c_f_ch2 = np.copy(f_ch2)
                        c_f_ch3 = np.copy(f_ch3)
                        c_f_ch4 = np.copy(f_ch4)
                        c_f_ch5 = np.copy(f_ch5)
                        c_f_ch6 = np.copy(f_ch6)
                        c_f_ch7 = np.copy(f_ch7)

                        F_ch0_abs = np.abs(f_ch0)
                        F_ch1_abs = np.abs(f_ch1)
                        F_ch2_abs = np.abs(f_ch2)
                        F_ch3_abs = np.abs(f_ch3)
                        F_ch4_abs = np.abs(f_ch4)
                        F_ch5_abs = np.abs(f_ch5)
                        F_ch6_abs = np.abs(f_ch6)
                        F_ch7_abs = np.abs(f_ch7)

                        # c_f_ch0 = F_ch0_abs
                        # c_f_ch1 = F_ch1_abs
                        # c_f_ch2 = F_ch2_abs
                        # c_f_ch3 = F_ch3_abs
                        # c_f_ch4 = F_ch4_abs
                        # c_f_ch5 = F_ch5_abs
                        # c_f_ch6 = F_ch6_abs
                        # c_f_ch7 = F_ch7_abs

                        F_ch0_abs_amp = F_ch0_abs / N * 2
                        F_ch1_abs_amp = F_ch1_abs / N * 2
                        F_ch2_abs_amp = F_ch2_abs / N * 2
                        F_ch3_abs_amp = F_ch3_abs / N * 2
                        F_ch4_abs_amp = F_ch4_abs / N * 2
                        F_ch5_abs_amp = F_ch5_abs / N * 2
                        F_ch6_abs_amp = F_ch6_abs / N * 2
                        F_ch7_abs_amp = F_ch7_abs / N * 2

                        F_ch0_abs_amp[ 0 ] = F_ch0_abs_amp[ 0 ] / 2
                        F_ch1_abs_amp[ 0 ] = F_ch1_abs_amp[ 0 ] / 2
                        F_ch2_abs_amp[ 0 ] = F_ch2_abs_amp[ 0 ] / 2
                        F_ch3_abs_amp[ 0 ] = F_ch3_abs_amp[ 0 ] / 2
                        F_ch4_abs_amp[ 0 ] = F_ch4_abs_amp[ 0 ] / 2
                        F_ch5_abs_amp[ 0 ] = F_ch5_abs_amp[ 0 ] / 2
                        F_ch6_abs_amp[ 0 ] = F_ch6_abs_amp[ 0 ] / 2
                        F_ch7_abs_amp[ 0 ] = F_ch7_abs_amp[ 0 ] / 2

                        fq = np.linspace(0, 1.0 / dt, N)

                        # print(fq.shape)

                        c_f_ch0[ (fq < fc) ] = 0
                        c_f_ch1[ (fq < fc) ] = 0
                        c_f_ch2[ (fq < fc) ] = 0
                        c_f_ch3[ (fq < fc) ] = 0
                        c_f_ch4[ (fq < fc) ] = 0
                        c_f_ch5[ (fq < fc) ] = 0
                        c_f_ch6[ (fq < fc) ] = 0
                        c_f_ch7[ (fq < fc) ] = 0
                        # print(c_f_ch0)

                        # print(c_f_ch0.shape)

                        c_0 = 0
                        c_1 = 0
                        c_2 = 0
                        c_3 = 0
                        c_4 = 0
                        c_5 = 0
                        c_6 = 0
                        c_7 = 0

                        for hm in range(60, 500, 60):
                            for p in range(1, 5):
                                if hm == 60:
                                    hm -= 13
                                c_0 += c_f_ch0[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch0[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_1 += c_f_ch1[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch1[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_2 += c_f_ch2[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch2[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_3 += c_f_ch3[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch3[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_4 += c_f_ch4[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch4[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_5 += c_f_ch5[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch5[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_6 += c_f_ch6[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch6[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                c_7 += c_f_ch7[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch7[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
                                # print("c_f_ch0[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]")
                                # print(c_f_ch0[ (fq > (hm - p)-1) & (fq < (hm + p)+1) ])

                            c_0 = c_0 / 10
                            c_1 = c_1 / 10
                            c_2 = c_2 / 10
                            c_3 = c_3 / 10
                            c_4 = c_4 / 10
                            c_5 = c_5 / 10
                            c_6 = c_6 / 10
                            c_7 = c_7 / 10
                            # print(c_0)

                            c_f_ch0[ (fq > hm - 5) & (fq < hm + 5) ] = c_0
                            c_f_ch1[ (fq > hm - 5) & (fq < hm + 5) ] = c_1
                            c_f_ch2[ (fq > hm - 5) & (fq < hm + 5) ] = c_2
                            c_f_ch3[ (fq > hm - 5) & (fq < hm + 5) ] = c_3
                            c_f_ch4[ (fq > hm - 5) & (fq < hm + 5) ] = c_4
                            c_f_ch5[ (fq > hm - 5) & (fq < hm + 5) ] = c_5
                            c_f_ch6[ (fq > hm - 5) & (fq < hm + 5) ] = c_6
                            c_f_ch7[ (fq > hm - 5) & (fq < hm + 5) ] = c_7

                        x = np.linspace(0, 1 / dt, len(c_f_ch0))
                        plt.plot(x, c_f_ch0, c="r", alpha=0.7, label='filtered')
                        plt.savefig(path.preprocessing_abs + "/" + i + "/" + d + "/" + s
                                    + "/" + o + "/" + j + "/plt3_noize_remove.png")
                        plt.close()
                        # plt.show()

                        c_F_ch0_abs = np.abs(c_f_ch0)
                        c_F_ch1_abs = np.abs(c_f_ch1)
                        c_F_ch2_abs = np.abs(c_f_ch2)
                        c_F_ch3_abs = np.abs(c_f_ch3)
                        c_F_ch4_abs = np.abs(c_f_ch4)
                        c_F_ch5_abs = np.abs(c_f_ch5)
                        c_F_ch6_abs = np.abs(c_f_ch6)
                        c_F_ch7_abs = np.abs(c_f_ch7)

                        c_F_ch0_abs_amp = c_F_ch0_abs / N * 2
                        c_F_ch1_abs_amp = c_F_ch1_abs / N * 2
                        c_F_ch2_abs_amp = c_F_ch2_abs / N * 2
                        c_F_ch3_abs_amp = c_F_ch3_abs / N * 2
                        c_F_ch4_abs_amp = c_F_ch4_abs / N * 2
                        c_F_ch5_abs_amp = c_F_ch5_abs / N * 2
                        c_F_ch6_abs_amp = c_F_ch6_abs / N * 2
                        c_F_ch7_abs_amp = c_F_ch7_abs / N * 2

                        c_F_ch0_abs_amp[ 0 ] = c_F_ch0_abs_amp[ 0 ] / 2
                        c_F_ch1_abs_amp[ 0 ] = c_F_ch1_abs_amp[ 0 ] / 2
                        c_F_ch2_abs_amp[ 0 ] = c_F_ch2_abs_amp[ 0 ] / 2
                        c_F_ch3_abs_amp[ 0 ] = c_F_ch3_abs_amp[ 0 ] / 2
                        c_F_ch4_abs_amp[ 0 ] = c_F_ch4_abs_amp[ 0 ] / 2
                        c_F_ch5_abs_amp[ 0 ] = c_F_ch5_abs_amp[ 0 ] / 2
                        c_F_ch6_abs_amp[ 0 ] = c_F_ch6_abs_amp[ 0 ] / 2
                        c_F_ch7_abs_amp[ 0 ] = c_F_ch7_abs_amp[ 0 ] / 2

                        c_f_ch0_ifft = np.fft.ifft(c_f_ch0)
                        c_f_ch1_ifft = np.fft.ifft(c_f_ch1)
                        c_f_ch2_ifft = np.fft.ifft(c_f_ch2)
                        c_f_ch3_ifft = np.fft.ifft(c_f_ch3)
                        c_f_ch4_ifft = np.fft.ifft(c_f_ch4)
                        c_f_ch5_ifft = np.fft.ifft(c_f_ch5)
                        c_f_ch6_ifft = np.fft.ifft(c_f_ch6)
                        c_f_ch7_ifft = np.fft.ifft(c_f_ch7)


                        c_f_ch0_ifft_real = c_f_ch0_ifft.real * 2
                        c_f_ch1_ifft_real = c_f_ch1_ifft.real * 2
                        c_f_ch2_ifft_real = c_f_ch2_ifft.real * 2
                        c_f_ch3_ifft_real = c_f_ch3_ifft.real * 2
                        c_f_ch4_ifft_real = c_f_ch4_ifft.real * 2
                        c_f_ch5_ifft_real = c_f_ch5_ifft.real * 2
                        c_f_ch6_ifft_real = c_f_ch6_ifft.real * 2
                        c_f_ch7_ifft_real = c_f_ch7_ifft.real * 2

                        x = np.linspace(0, 1 / dt, len(c_f_ch0_ifft_real))
                        plt.plot(x, c_f_ch0_ifft_real, c="r", alpha=0.7, label='filtered')
                        plt.savefig(path.preprocessing_abs + "/" + i + "/" + d + "/" + s
                                    + "/" + o + "/" + j + "/plt4_after_ifft.png")
                        plt.close()
                        # plt.show()

                        c_f_ch0_ifft_real = abs(c_f_ch0_ifft_real)
                        c_f_ch1_ifft_real = abs(c_f_ch1_ifft_real)
                        c_f_ch2_ifft_real = abs(c_f_ch2_ifft_real)
                        c_f_ch3_ifft_real = abs(c_f_ch3_ifft_real)
                        c_f_ch4_ifft_real = abs(c_f_ch4_ifft_real)
                        c_f_ch5_ifft_real = abs(c_f_ch5_ifft_real)
                        c_f_ch6_ifft_real = abs(c_f_ch6_ifft_real)
                        c_f_ch7_ifft_real = abs(c_f_ch7_ifft_real)

                        x = np.linspace(0, 1 / dt, len(c_f_ch0_ifft_real))
                        plt.plot(x, c_f_ch0_ifft_real, c="r", alpha=0.7, label='filtered')
                        plt.savefig(path.preprocessing_abs + "/" + i + "/" + d + "/" + s
                                    + "/" + o + "/" + j + "/plt5_abs.png")
                        plt.close()
                        # plt.show()

                        ch0 = np.convolve(c_f_ch0_ifft_real, b, mode='same')
                        ch1 = np.convolve(c_f_ch1_ifft_real, b, mode='same')
                        ch2 = np.convolve(c_f_ch2_ifft_real, b, mode='same')
                        ch3 = np.convolve(c_f_ch3_ifft_real, b, mode='same')
                        ch4 = np.convolve(c_f_ch4_ifft_real, b, mode='same')
                        ch5 = np.convolve(c_f_ch5_ifft_real, b, mode='same')
                        ch6 = np.convolve(c_f_ch6_ifft_real, b, mode='same')
                        ch7 = np.convolve(c_f_ch7_ifft_real, b, mode='same')

                        x = np.linspace(0, 1 / dt, len(ch0))
                        plt.plot(x, ch0, c="r", alpha=0.7, label='filtered')
                        plt.savefig(path.preprocessing_abs + "/" + i + "/" + d + "/" + s
                                    + "/" + o + "/" + j + "/plt6_finish.png")
                        plt.close()
                        # plt.show()

                        y1 = pd.DataFrame(ch0)
                        y2 = pd.DataFrame(ch1)
                        y3 = pd.DataFrame(ch2)
                        y4 = pd.DataFrame(ch3)
                        y5 = pd.DataFrame(ch4)
                        y6 = pd.DataFrame(ch5)
                        y7 = pd.DataFrame(ch6)
                        y8 = pd.DataFrame(ch7)


                        nf = open(
                            path.preprocessing_abs + "/" + i + "/" + d + "/" + s + "/" + o + "/" + j + "/" + l + ".CSV",
                            'w')
                        dataWriter = csv.writer(nf)

                        new_data = pd.concat([ y1, y2, y3, y4, y5, y6, y7, y8 ], axis=1)
                        # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7], axis=1)
                        # new_data = pd.concat([ y1, y2, y3, y4], axis=1)

                        # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                        new_data.columns = new_data.iloc[ 0, : ]
                        new_data.index = new_data.iloc[ :, 0 ]
                        new_data = new_data.iloc[ 1:255, 1:8 ]
                        # new_data = new_data.iloc[1:129, 1:7]
                        # new_data = new_data.iloc[ 1:255, 1:4]

                        new_data.to_csv(
                            path.preprocessing_abs + "/" + i + "/" + d + "/" + s + "/" + o + "/" + j + "/" + l + ".CSV")
                        print("preprocessing_abs/" + i + "/" + d + "/" + s + "/" + o + "/" + j + "/" + l + ".CSV")
                        print(
                            "////////////////////////////////////////////////////////////////////////////////" +
                            "//////////////////////")

                        nf.close()


