import pandas as pd
from sklearn.decomposition import PCA
from app import path
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import csv

N = int((path.data_long / 8) - 1)
dt = 0.001
fc = 20
# hm = 60
train_x = []
train_y = []



# ファイルの読み込み
for i in path.new_subject:
    for d in path.new_day:
        for s in path.sets:
            for j in path.mix_char2:
                for o in path.cutplusmix_time:
                    for l in path.time:
                        file_path = path.cut_plusmix_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        df = pd.read_csv(file_path)
                        f_df = np.fft.fft(df)
                        print("cut_plustmix_new/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "cut.CSV")
                        # print(df)

                        ch0 = list(df.iloc[:, 0].values)
                        ch1 = list(df.iloc[:, 1].values)
                        ch2 = list(df.iloc[:, 2].values)
                        ch3 = list(df.iloc[:, 3].values)
                        ch4 = list(df.iloc[:, 4].values)
                        ch5 = list(df.iloc[:, 5].values)
                        ch6 = list(df.iloc[:, 6].values)
                        ch7 = list(df.iloc[:, 7].values)

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


                        c_f_ch0 = F_ch0_abs
                        c_f_ch1 = F_ch1_abs
                        c_f_ch2 = F_ch2_abs
                        c_f_ch3 = F_ch3_abs
                        c_f_ch4 = F_ch4_abs
                        c_f_ch5 = F_ch5_abs
                        c_f_ch6 = F_ch6_abs
                        c_f_ch7 = F_ch7_abs

                        F_ch0_abs_amp = F_ch0_abs / N * 2
                        F_ch1_abs_amp = F_ch1_abs / N * 2
                        F_ch2_abs_amp = F_ch2_abs / N * 2
                        F_ch3_abs_amp = F_ch3_abs / N * 2
                        F_ch4_abs_amp = F_ch4_abs / N * 2
                        F_ch5_abs_amp = F_ch5_abs / N * 2
                        F_ch6_abs_amp = F_ch6_abs / N * 2
                        F_ch7_abs_amp = F_ch7_abs / N * 2

                        F_ch0_abs_amp[0] = F_ch0_abs_amp[0] / 2
                        F_ch1_abs_amp[0] = F_ch1_abs_amp[0] / 2
                        F_ch2_abs_amp[0] = F_ch2_abs_amp[0] / 2
                        F_ch3_abs_amp[0] = F_ch3_abs_amp[0] / 2
                        F_ch4_abs_amp[0] = F_ch4_abs_amp[0] / 2
                        F_ch5_abs_amp[0] = F_ch5_abs_amp[0] / 2
                        F_ch6_abs_amp[0] = F_ch6_abs_amp[0] / 2
                        F_ch7_abs_amp[0] = F_ch7_abs_amp[0] / 2

                        fq = np.linspace(0, 1.0 / dt, N)

                        # print(fq.shape)

                        c_f_ch0[(fq < fc)] = 0
                        c_f_ch1[(fq < fc)] = 0
                        c_f_ch2[(fq < fc)] = 0
                        c_f_ch3[(fq < fc)] = 0
                        c_f_ch4[(fq < fc)] = 0
                        c_f_ch5[(fq < fc)] = 0
                        c_f_ch6[(fq < fc)] = 0
                        c_f_ch7[(fq < fc)] = 0
                        # print(c_f_ch0)

                        print(c_f_ch0.shape)

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
                                c_0 += c_f_ch0[(fq > (hm - p) - 2) & (fq < (hm - p) + 2)] + \
                                       c_f_ch0[(fq > (hm - p) - 2) & (fq < (hm - p) + 2)]
                                c_1 += c_f_ch1[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ] + \
                                       c_f_ch1[(fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]
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
                                print("c_f_ch0[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ]")
                                print(c_f_ch0[ (fq > (hm - p) - 2) & (fq < (hm - p) + 2) ])



                            c_0 = c_0 / 10
                            c_1 = c_1 / 10
                            c_2 = c_2 / 10
                            c_3 = c_3 / 10
                            c_4 = c_4 / 10
                            c_5 = c_5 / 10
                            c_6 = c_6 / 10
                            c_7 = c_7 / 10
                            print(c_0)

                            c_f_ch0[(fq > hm - 5) & (fq < hm + 5)] = c_0
                            c_f_ch1[(fq > hm - 5) & (fq < hm + 5)] = c_1
                            c_f_ch2[(fq > hm - 5) & (fq < hm + 5)] = c_2
                            c_f_ch3[(fq > hm - 5) & (fq < hm + 5)] = c_3
                            c_f_ch4[(fq > hm - 5) & (fq < hm + 5)] = c_4
                            c_f_ch5[(fq > hm - 5) & (fq < hm + 5)] = c_5
                            c_f_ch6[(fq > hm - 5) & (fq < hm + 5)] = c_6
                            c_f_ch7[(fq > hm - 5) & (fq < hm + 5)] = c_7

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

                        c_F_ch0_abs_amp[0] = c_F_ch0_abs_amp[0] / 2
                        c_F_ch1_abs_amp[0] = c_F_ch1_abs_amp[0] / 2
                        c_F_ch2_abs_amp[0] = c_F_ch2_abs_amp[0] / 2
                        c_F_ch3_abs_amp[0] = c_F_ch3_abs_amp[0] / 2
                        c_F_ch4_abs_amp[0] = c_F_ch4_abs_amp[0] / 2
                        c_F_ch5_abs_amp[0] = c_F_ch5_abs_amp[0] / 2
                        c_F_ch6_abs_amp[0] = c_F_ch6_abs_amp[0] / 2
                        c_F_ch7_abs_amp[0] = c_F_ch7_abs_amp[0] / 2

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



                        nf = open(path.fft_mix_15_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "fft.CSV", 'w')
                        dataWriter = csv.writer(nf)

                        fft0 = len(c_f_ch0)
                        fft1 = len(c_f_ch1)
                        fft2 = len(c_f_ch2)
                        fft3 = len(c_f_ch3)
                        fft4 = len(c_f_ch4)
                        fft5 = len(c_f_ch5)
                        fft6 = len(c_f_ch6)
                        fft7 = len(c_f_ch7)

                        c_f_ch0_p = pd.DataFrame(c_f_ch0)
                        c_f_ch1_p = pd.DataFrame(c_f_ch1)
                        c_f_ch2_p = pd.DataFrame(c_f_ch2)
                        c_f_ch3_p = pd.DataFrame(c_f_ch3)
                        c_f_ch4_p = pd.DataFrame(c_f_ch4)
                        c_f_ch5_p = pd.DataFrame(c_f_ch5)
                        c_f_ch6_p = pd.DataFrame(c_f_ch6)
                        c_f_ch7_p = pd.DataFrame(c_f_ch7)

                        # データの切り出し
                        y1 = c_f_ch0_p.iloc[0:int(fft0 / 2), :]
                        y2 = c_f_ch1_p.iloc[0:int(fft1 / 2), :]
                        y3 = c_f_ch2_p.iloc[0:int(fft2 / 2), :]
                        y4 = c_f_ch3_p.iloc[0:int(fft3 / 2), :]
                        y5 = c_f_ch4_p.iloc[0:int(fft4 / 2), :]
                        y6 = c_f_ch5_p.iloc[0:int(fft5 / 2), :]
                        y7 = c_f_ch6_p.iloc[0:int(fft6 / 2), :]
                        y8 = c_f_ch7_p.iloc[0:int(fft7 / 2), :]

                        print(y1.shape)



                        new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7, y8], axis=1)
                        # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7], axis=1)
                        # new_data = pd.concat([ y1, y2, y3, y4], axis=1)

                        # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                        new_data.columns = new_data.iloc[0, :]
                        new_data.index = new_data.iloc[:, 0]
                        new_data = new_data.iloc[1:129, 1:8]
                        # new_data = new_data.iloc[1:129, 1:7]
                        # new_data = new_data.iloc[ 1:129, 1:4]

                        new_data.to_csv(path.fft_mix_15_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "fft.CSV")
                        print("fft/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "fft.CSV")

                        nf.close()
# plt.plot(fq, c_F_ch0_abs_amp)
# plt.plot(fq,c_F_ch0_abs, c='r')
# plt.plot(fq,y,label='original')
# plt.axis([0, 1.0 / dt / 2, 0, max(F_ch0_abs_amp) * 1.5])
x2 = np.linspace(0,1/dt,len(c_f_ch0))
plt.plot(x2, c_f_ch0, c="r", alpha=0.7, label='filtered')
print(len(c_F_ch0_abs_amp)/2)
plt.show()
