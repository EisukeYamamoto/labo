import numpy as np
import matplotlib.pyplot as plt
from app import ica
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from app import path
from sklearn import linear_model

import csv

N = path.data_long
train_x = []
clf = linear_model.LinearRegression()

def write_plot(ch_list):
    # print("test/" + d + "/" + i + "/" + j + "/" + l + ".CSV/")
    len_num = len(ch_list[0])
    x = np.linspace(0, len_num, len_num)
    y0 = ch_list[0]
    y1 = ch_list[1]
    y2 = ch_list[2]
    y3 = ch_list[3]
    y4 = ch_list[4]
    y5 = ch_list[5]
    y6 = ch_list[6]

    labels0 = [-0.5, 0.5]
    labels1 = [-0.5, 0.5]
    labels2 = [-0.5, 0.5]
    labels3 = [-0.5, 0.5]
    labels4 = [-0.5, 0.5]
    labels5 = [-0.5, 0.5]
    labels6 = [-0.5, 0.5]

    l0,l1,l2,l3,l4,l5,l6 = "ch0","ch1","ch2","ch3","ch4","ch5","ch6"
    o0,o1,o2,o3,o4,o5,o6 = 1,3,5,7,9,11,13

    yticks0 = [la + o0 for la in labels0]
    yticks1 = [la + o1 for la in labels1]
    yticks2 = [la + o2 for la in labels2]
    yticks3 = [la + o3 for la in labels3]
    yticks4 = [la + o4 for la in labels4]
    yticks5 = [la + o5 for la in labels5]
    yticks6 = [la + o6 for la in labels6]

    ytls = labels0+labels1+labels2+labels3+labels4+labels5+labels6
    ytks = yticks0+yticks1+yticks2+yticks3+yticks4+yticks5+yticks6
    plt.figure(figsize=(6, 5), facecolor="w")
    yo0 = list(map(lambda x: x + o6, y0))
    yo1 = list(map(lambda x: x + o5, y1))
    yo2 = list(map(lambda x: x + o4, y2))
    yo3 = list(map(lambda x: x + o3, y3))
    yo4 = list(map(lambda x: x + o2, y4))
    yo5 = list(map(lambda x: x + o1, y5))
    yo6 = list(map(lambda x: x + o0, y6))


    plt.plot(x, yo0 , color="b", label = l0)
    plt.plot(x, yo1 , color="r", label = l1)
    plt.plot(x, yo2 , color="g", label = l2)
    plt.plot(x, yo3 , color="c", label = l3)
    plt.plot(x, yo4 , color="m", label = l4)
    plt.plot(x, yo5 , color="y", label = l5)
    plt.plot(x, yo6 , color="k", label = l6)

    plt.ylim(o0-2,o6+2)
    plt.yticks(ytks)
    plt.axes().set_yticklabels(ytls)
    plt.legend(loc="upper right", fontsize=8)

    labs = plt.axes().get_yticklabels()





    #df.plot(ylim=(-0.5,0.5),color="blue",subplots=True,layout=(8, 1),legend=False,fontsize=10)
    #plt.gca().spines['right'].set_visible(False)
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['bottom'].set_visible(False)
    #plt.gca().yaxis.set_ticks_position('left')
    #plt.subplots_adjust(wspace=0.5, hspace=1)
    #plt.tight_layout()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.tick_params(bottom=False,left=False,right=False,top=False)


    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.savefig(path.png_mix + "/" + d + "/" + i + "/" + j + "/" + o + "/" +"/mix"+l)
    plt.show()
    plt.close('all')


# 複合動作
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.mix_only:
            for l in path.times:
                file_path = path.ifft_mix + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                df1 = pd.read_csv(file_path)

                mix_ch0 = list(df1.iloc[:, 0].values)
                mix_ch1 = list(df1.iloc[:, 1].values)
                mix_ch2 = list(df1.iloc[:, 2].values)
                mix_ch3 = list(df1.iloc[:, 3].values)
                mix_ch4 = list(df1.iloc[:, 4].values)
                mix_ch5 = list(df1.iloc[:, 5].values)
                mix_ch6 = list(df1.iloc[:, 6].values)

                print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")

                csv_file = open(file_path, "r", encoding="ms932", errors="", newline="")
                f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)

                # nf = open(path.after_ica + "/" + d + "/" + i + "/" + j + "/" + l + "ica.CSV", 'w')
                # dataWriter = csv.writer(nf)

# 手首動作
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.tekubi_only:
            for l in path.times:
                file_path = path.ifft_mix + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                df2 = pd.read_csv(file_path)

                tekubi_ch0 = list(df2.iloc[:, 0].values)
                tekubi_ch1 = list(df2.iloc[:, 1].values)
                tekubi_ch2 = list(df2.iloc[:, 2].values)
                tekubi_ch3 = list(df2.iloc[:, 3].values)
                tekubi_ch4 = list(df2.iloc[:, 4].values)
                tekubi_ch5 = list(df2.iloc[:, 5].values)
                tekubi_ch6 = list(df2.iloc[:, 6].values)

                print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")


# 指動作
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.yubi_only:
            for l in path.times:
                file_path = path.ifft_mix + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                df3 = pd.read_csv(file_path)

                yubi_ch0 = list(df3.iloc[:, 0].values)
                yubi_ch1 = list(df3.iloc[:, 1].values)
                yubi_ch2 = list(df3.iloc[:, 2].values)
                yubi_ch3 = list(df3.iloc[:, 3].values)
                yubi_ch4 = list(df3.iloc[:, 4].values)
                yubi_ch5 = list(df3.iloc[:, 5].values)
                yubi_ch6 = list(df3.iloc[:, 6].values)

                print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")







# 独立成分の数＝2
decomposer = FastICA(n_components = 2)

# データの平均を計算
M1 = np.mean(df1, axis=1)[:, np.newaxis]
M2 = np.mean(df2, axis=1)[:, np.newaxis]

data1 = df1 - M1
data2 = df2 - M2

data1_p = pd.DataFrame(data1)

data1_1 = list(data1_p.iloc[:, 0].values)
# 平均0としたデータに対して、独立成分分析を実施
S_ = decomposer.fit_transform(data1_1)

# 独立成分ベクトルを取得(D次元 x 独立成分数)
# S_1 = decomposer.transform(data1)

# 混合行列の計算（データ数 x 独立性分数）
W = decomposer.mixing_

# 混合行列と独立成分から元の信号dataを復元
# X1 = np.dot(S_1, W.T)
# X1 += M1

# X2 = np.dot(S_1, W.T)
# X2 += M2

# 混合行列の擬似逆行列を取得
# W_inv = decomposer.components_

print(S_.shape)

X_p = pd.DataFrame(S_)
Y_p = pd.DataFrame(S_)
S_p = pd.DataFrame(S_)


clf.fit(df3, Y_p)

print(clf.coef_)
print(clf.score(df3, Y_p))


afterX0 = list(X_p.iloc[:, 0].values)
afterX1 = list(X_p.iloc[:, 1].values)
afterX2 = list(X_p.iloc[:, 2].values)
afterX3 = list(X_p.iloc[:, 3].values)
afterX4 = list(X_p.iloc[:, 4].values)
afterX5 = list(X_p.iloc[:, 5].values)
afterX6 = list(X_p.iloc[:, 6].values)

afterY0 = list(Y_p.iloc[:, 0].values)
afterY1 = list(Y_p.iloc[:, 1].values)
afterY2 = list(Y_p.iloc[:, 2].values)
afterY3 = list(Y_p.iloc[:, 3].values)
afterY4 = list(Y_p.iloc[:, 4].values)
afterY5 = list(Y_p.iloc[:, 5].values)
afterY6 = list(Y_p.iloc[:, 6].values)

S0 = list(S_p.iloc[:, 0].values)
S1 = list(S_p.iloc[:, 1].values)
S2 = list(S_p.iloc[:, 2].values)
S3 = list(S_p.iloc[:, 3].values)
S4 = list(S_p.iloc[:, 4].values)
S5 = list(S_p.iloc[:, 5].values)
S6 = list(S_p.iloc[:, 6].values)

# plt.plot(S0)
# plt.show()
# plt.plot(S1)
# plt.show()

# plt.plot(W_inv)
# plt.show()
print(data1.T.shape)
# print(S_1.shape)
# print(W_inv)

tekubi_ch_list = [tekubi_ch0, tekubi_ch1, tekubi_ch2, tekubi_ch3, tekubi_ch4, tekubi_ch5, tekubi_ch6]
mix_ch_list = [mix_ch0, mix_ch1, mix_ch2, mix_ch3, mix_ch4, mix_ch5, mix_ch6]
yubi_ch_list = [yubi_ch0, yubi_ch1, yubi_ch2, yubi_ch3, yubi_ch4, yubi_ch5, yubi_ch6]
afterX_list = [afterX0, afterX1, afterX2, afterX3, afterX4, afterX5, afterX6]
afterY_list = [afterY0, afterY1, afterY2, afterY3, afterY4, afterY5, afterY6]
# S_list = [S0, S1, S2, S3, S4, S5, S6]

write_plot(mix_ch_list)
write_plot(afterX_list)
write_plot(tekubi_ch_list)
# write_plot(S_list)
# write_plot(afterY_list)
write_plot(yubi_ch_list)