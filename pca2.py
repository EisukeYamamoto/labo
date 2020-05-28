"""
主成分分析
"""

import pandas as pd
from sklearn.decomposition import PCA, kernel_pca
from app import path
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 127
dt = 0.01
fc = 25

# 外れ値の関数
def outlier_iqr(df):

    for i in range(len(df.columns)):
        # 四分位数
        a = X_pca.iloc[:, i]

        Q1 = a.quantile(.25)
        Q3 = a.quantile(.75)
        # 四分位数
        q = Q3 - Q1

        # 外れ値の基準点
        outlier_min = Q1 - q * 1.5
        outlier_max = Q3 + q * 1.5

        # 範囲から外れている値を除く
        a[a < outlier_min] = None
        a[a > outlier_max] = None

    return df


train_x = []
train_y = []

# ファイルの読み込み
for d in path.day:
    for i in path.subject:
        for j in path.char:
            for o in path.cut_time:
                for l in path.time:
                    file_path = path.cut8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV"
                    df = pd.read_csv(file_path)
                    # print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                    # print(df)

                    ch0 = list(df.iloc[:, 0].values)
                    ch1 = list(df.iloc[:, 1].values)
                    ch2 = list(df.iloc[:, 2].values)
                    ch3 = list(df.iloc[:, 3].values)
                    ch4 = list(df.iloc[:, 4].values)
                    ch5 = list(df.iloc[:, 5].values)
                    ch6 = list(df.iloc[:, 6].values)
                    # ch7 = list(df.iloc[:, 7].values)

                    n_ch0 = np.array(ch0)
                    n_ch1 = np.array(ch1)
                    n_ch2 = np.array(ch2)
                    n_ch3 = np.array(ch3)
                    n_ch4 = np.array(ch4)
                    n_ch5 = np.array(ch5)
                    n_ch6 = np.array(ch6)

                    # フーリエ変換

                    f_ch0 = np.fft.fft(n_ch0)
                    f_ch1 = np.fft.fft(n_ch1)
                    f_ch2 = np.fft.fft(n_ch2)
                    f_ch3 = np.fft.fft(n_ch3)
                    f_ch4 = np.fft.fft(n_ch4)
                    f_ch5 = np.fft.fft(n_ch5)
                    f_ch6 = np.fft.fft(n_ch6)

                    fq = np.linspace(0, 1.0 / dt, N)

                    f_ch0[(fq > fc)] = 0
                    f_ch1[(fq > fc)] = 0
                    f_ch2[(fq > fc)] = 0
                    f_ch3[(fq > fc)] = 0
                    f_ch4[(fq > fc)] = 0
                    f_ch5[(fq > fc)] = 0
                    f_ch6[(fq > fc)] = 0

                    F_ch0_abs = np.abs(f_ch0)
                    F_ch1_abs = np.abs(f_ch1)
                    F_ch2_abs = np.abs(f_ch2)
                    F_ch3_abs = np.abs(f_ch3)
                    F_ch4_abs = np.abs(f_ch4)
                    F_ch5_abs = np.abs(f_ch5)
                    F_ch6_abs = np.abs(f_ch6)

                    F_ch0_abs_amp = F_ch0_abs / N * 2
                    F_ch1_abs_amp = F_ch1_abs / N * 2
                    F_ch2_abs_amp = F_ch2_abs / N * 2
                    F_ch3_abs_amp = F_ch3_abs / N * 2
                    F_ch4_abs_amp = F_ch4_abs / N * 2
                    F_ch5_abs_amp = F_ch5_abs / N * 2
                    F_ch6_abs_amp = F_ch6_abs / N * 2

                    F_ch0_abs_amp[0] = F_ch0_abs_amp[0] / 2
                    F_ch1_abs_amp[0] = F_ch1_abs_amp[0] / 2
                    F_ch2_abs_amp[0] = F_ch2_abs_amp[0] / 2
                    F_ch3_abs_amp[0] = F_ch3_abs_amp[0] / 2
                    F_ch4_abs_amp[0] = F_ch4_abs_amp[0] / 2
                    F_ch5_abs_amp[0] = F_ch5_abs_amp[0] / 2
                    F_ch6_abs_amp[0] = F_ch6_abs_amp[0] / 2

                    f_ch0_ifft = np.fft.ifft(f_ch0)
                    f_ch1_ifft = np.fft.ifft(f_ch1)
                    f_ch2_ifft = np.fft.ifft(f_ch2)
                    f_ch3_ifft = np.fft.ifft(f_ch3)
                    f_ch4_ifft = np.fft.ifft(f_ch4)
                    f_ch5_ifft = np.fft.ifft(f_ch5)
                    f_ch6_ifft = np.fft.ifft(f_ch6)

                    f_ch0_ifft_real = f_ch0_ifft.real * 2
                    f_ch1_ifft_real = f_ch1_ifft.real * 2
                    f_ch2_ifft_real = f_ch2_ifft.real * 2
                    f_ch3_ifft_real = f_ch3_ifft.real * 2
                    f_ch4_ifft_real = f_ch4_ifft.real * 2
                    f_ch5_ifft_real = f_ch5_ifft.real * 2
                    f_ch6_ifft_real = f_ch6_ifft.real * 2

                    feature_value = []
                    feature_value.extend(f_ch0_ifft_real)
                    feature_value.extend(f_ch1_ifft_real)
                    feature_value.extend(f_ch2_ifft_real)
                    feature_value.extend(f_ch3_ifft_real)
                    feature_value.extend(f_ch4_ifft_real)
                    feature_value.extend(f_ch5_ifft_real)
                    feature_value.extend(f_ch6_ifft_real)

                    # feature_value.extend(ch0)
                    # feature_value.extend(ch1)
                    # feature_value.extend(ch2)
                    # feature_value.extend(ch3)
                    # feature_value.extend(ch4)
                    # feature_value.extend(ch5)
                    # feature_value.extend(ch6)
                    # feature_value.extend(ch7)

                    train_x.append(feature_value)
                    # numpy.savetxt(path.pca + "/" + d + "/" + i + "/" + "data.csv", train_x, delimiter=',')

                    if (j == '指'):
                        train_y.append('指')
                    elif (j == '手首'):
                        train_y.append('手首')


# 標準化
sc = StandardScaler()
x = sc.fit_transform(train_x)

# 約8000次元から8次元に圧縮
# n_componentsで次元変更
pca = PCA(n_components=2)
# 標準化したxデータをPCAで次元圧縮
X_pca = pd.DataFrame(pca.fit_transform(train_x))

# kpca = kernel_pca(n_components=2)
# X_kpca = pd.DataFrame(kpca.fit_transform(train_x))

#外れ値
z = outlier_iqr(X_pca)

# print(X_pca)
print(z)
# 色の指定
# colors = [plt.cm.nipy_spectral(i / 5., 1) for i in range(5)]
# colors = ['red','green','blue','yellow','black']

#二次元plot
plt.figure(figsize=(6, 6))

# for c,x_pca in zip(colors,z):
#for x_pca in z:
#    plt.scatter(X_pca.iloc[0:10,x_pca], X_pca.iloc[0:10,x_pca+1],c='red')
#    plt.scatter(X_pca.iloc[10:20,x_pca], X_pca.iloc[10:20,x_pca+ 1],c='blue')
#    plt.scatter(X_pca.iloc[20:30,x_pca], X_pca.iloc[20:30,x_pca+ 1],c='green')
#    plt.scatter(X_pca.iloc[30:40,x_pca], X_pca.iloc[30:40,x_pca+ 1],c='yellow')
#    plt.scatter(X_pca.iloc[40:50,x_pca], X_pca.iloc[40:50,x_pca+ 1],c='black')
X_0 = z.iloc[0:80,0]
X_1 = z.iloc[80:160,0]
Y_0 = z.iloc[0:80,1]
Y_1 = z.iloc[80:160,1]

plt.scatter(X_0,Y_0,c='red')
plt.scatter(X_1,Y_1,c='blue')

plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
# plt.xlim([-30,5])
# plt.ylim([-15,10])
# 保存時ファイル名変更
# plt.savefig(path.pca+"/1day"+"/sub_A"+"/pca_plot_clear")
plt.show()

print('各主成分の寄与率:', pca.explained_variance_ratio_)
print('寄与率の累積:', sum(pca.explained_variance_ratio_))

# print(pca.components_)

# csvファイルに出力
# numpy.savetxt(path.pca + "/" + d + "/" + i + "/" + "pca2.csv",pca.components_,delimiter=',')
