import matplotlib.pyplot as plt
# from app import ica
import pandas as pd
import numpy as np
import math
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from app import path
from sklearn import linear_model
from sklearn import preprocessing
import ica_SICA

import csv

clf0 = linear_model.LinearRegression()
clf1 = linear_model.LinearRegression()
clf2 = linear_model.LinearRegression()
clf3 = linear_model.LinearRegression()
clf4 = linear_model.LinearRegression()
clf5 = linear_model.LinearRegression()
clf6 = linear_model.LinearRegression()
clf7 = linear_model.LinearRegression()
sscaler = preprocessing.StandardScaler()
SICA = ica_SICA.SICA()
# mm = preprocessing.minmax_scale()

debug = 3  # 0:プロット無し, 1:各チャンネルごとの分離比較, 2:チャンネルを一式表示, 3:一式表示
plot_flg = 0  # 0:保存しない, 1:保存する
fft_flg = 1  # 0:時系列データ, 1:パワースペクトル　2:長さ二倍
clf_flg = 1  # 重回帰分析の目的関数  1:手首 2:複合動作 3:指
x_flg = 1  # 1:複合と複合2　2:複合と手首　3:複合と指


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


def center(X):
    X = np.array(X)

    mean = X.mean(axis=1, keepdims=True)

    return X - mean

def center2(X):
    X = np.array(X)

    mean = X.mean(axis=0, keepdims=True)

    return X - mean

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    loop = [0]
    print("components_nr")
    print(components_nr)

    for i in range(components_nr):
        w = np.random.rand(components_nr)
        # print(w)
        # print("///////////////////////////////")
        for j in range(iterations):
            w_new = calculate_new_w(w, X)
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new
            # print(w)

            if distance < tolerance:
                print("OK")
                # print(distance)
                break
            # print(distance)
            loop.append(j+1)

        # print("///////////////////////////////")
        W[i, :] = w

    S = np.dot(W, X)
    print(W)

    return S


def plot_mixture_sources_predictions(X, original_sources, S, ch):
    fig = plt.figure()

    # if ch >= 2:
    #     ch += 1

    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    plt.subplot(3, 1, 2)
    for o in original_sources:
        plt.plot(o)
    plt.title("real sources")
    plt.subplot(3, 1, 3)
    for a in S:
        plt.plot(a)
    plt.title("predicted sources")

    fig.tight_layout()
    # fig.subplots_adjust(hspace=5)

    # fig.suptitle(str(ch) + "ch")

    if plot_flg == 1:
        if x_flg == 1:
            # plotpath1 = path.png_ica + "/" + d + "/" + i + "/複合+複合2/各チャンネルの比較/" + str(ch) + "ch" #
            plotpath1 = path.png_ica_new + "/" + i + "/" + d + "/" + s + "/" + "fastICA" + "/各チャンネルの比較/" + str(ch) + "ch"
        elif x_flg == 2:
            plotpath1 = path.png_ica + "/" + d + "/" + i + "/複合+手首/各チャンネルの比較/" + str(ch) + "ch"
        else:
            plotpath1 = path.png_ica + "/" + d + "/" + i + "/複合+指/各チャンネルの比較/" + str(ch) + "ch"
        plt.savefig(plotpath1)
        # plt.show()
    else: plt.show()


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        X += 0.002 * np.random.normal(size=X.shape)

    return X


def max_ch(l):
    l_min = min(l)
    l_max = max(l)
    return [l_max - l_min, l_min]


def min_max(l, r, min_):
    print(min_)
    return [(i - min_) / r for i in l]

def min_max2(l, r, min_):

    return [((i - min_) / r) * 1 - 0 for i in l]


def mapping(ch_list):
    len_num = len(ch_list[0])
    x = np.linspace(0, len_num, len_num)
    y0 = ch_list[0]
    y1 = ch_list[1]
    y2 = ch_list[2]
    y3 = ch_list[3]
    y4 = ch_list[4]
    y5 = ch_list[5]
    y6 = ch_list[6]
    y7 = ch_list[7]

    y0_max = max_ch(y0)
    y1_max = max_ch(y1)
    y2_max = max_ch(y2)
    y3_max = max_ch(y3)
    y4_max = max_ch(y4)
    y5_max = max_ch(y5)
    y6_max = max_ch(y6)
    y7_max = max_ch(y7)

    max_list = [y0_max[0], y1_max[0], y2_max[0], y3_max[0], y4_max[0], y5_max[0], y6_max[0], y7_max[0]]
    # max_list = [ y0_max[ 0 ], y1_max[ 0 ], y2_max[ 0 ], y3_max[ 0 ]]
    len_list = [y0_max[1], y1_max[1], y2_max[1], y3_max[1], y4_max[1], y5_max[1], y6_max[1], y7_max[1]]
    # len_list = [ y0_max[ 1 ], y1_max[ 1 ], y2_max[ 1 ], y3_max[ 1 ]]

    range_ = max(max_list)
    min_ = len_list[max_list.index(range_)]

    y0 = min_max(y0, range_, min(y0))
    y1 = min_max(y1, range_, min(y1))
    y2 = min_max(y2, range_, min(y2))
    y3 = min_max(y3, range_, min(y3))
    y4 = min_max(y4, range_, min(y4))
    y5 = min_max(y5, range_, min(y5))
    y6 = min_max(y6, range_, min(y6))
    y7 = min_max(y7, range_, min(y7))

    y0 = center2(y0)
    y1 = center2(y1)
    y2 = center2(y2)
    y3 = center2(y3)
    y4 = center2(y4)
    y5 = center2(y5)
    y6 = center2(y6)
    y7 = center2(y7)

    new_ch_list = [y0, y1, y2, y3, y4, y5, y6, y7]
    # new_ch_list = [ y0, y1, y2, y3]

    return new_ch_list

def mapping_1(ch_list1, ch_list2):
    # len_num = len(ch_list[0])
    # x = np.linspace(0, len_num, len_num)
    y0_0 = ch_list1[0]
    y0_1 = ch_list1[1]
    y0_2 = ch_list1[2]
    y0_3 = ch_list1[3]
    y0_4 = ch_list1[4]
    y0_5 = ch_list1[5]
    y0_6 = ch_list1[6]
    y0_7 = ch_list1[7]


    y1_0 = ch_list2[0]
    y1_1 = ch_list2[1]
    y1_2 = ch_list2[2]
    y1_3 = ch_list2[3]
    y1_4 = ch_list2[4]
    y1_5 = ch_list2[5]
    y1_6 = ch_list2[6]
    y1_7 = ch_list2[7]

    y0_0_max = max_ch(y0_0)
    y0_1_max = max_ch(y0_1)
    y0_2_max = max_ch(y0_2)
    y0_3_max = max_ch(y0_3)
    y0_4_max = max_ch(y0_4)
    y0_5_max = max_ch(y0_5)
    y0_6_max = max_ch(y0_6)
    y0_7_max = max_ch(y0_7)

    y1_0_max = max_ch(y1_0)
    y1_1_max = max_ch(y1_1)
    y1_2_max = max_ch(y1_2)
    y1_3_max = max_ch(y1_3)
    y1_4_max = max_ch(y1_4)
    y1_5_max = max_ch(y1_5)
    y1_6_max = max_ch(y1_6)
    y1_7_max = max_ch(y1_7)

    max_list = [y0_0_max[0], y0_1_max[0], y0_2_max[0], y0_3_max[0], y0_4_max[0], y0_5_max[0], y0_6_max[0], y0_7_max[0],
                y1_0_max[0], y1_1_max[0], y1_2_max[0], y1_3_max[0], y1_4_max[0], y1_5_max[0], y1_6_max[0], y1_7_max[0]]
    len_list = [y0_0_max[1], y0_1_max[1], y0_2_max[1], y0_3_max[1], y0_4_max[1], y0_5_max[1], y0_6_max[1], y0_7_max[1],
                y1_0_max[1], y1_1_max[1], y1_2_max[1], y1_3_max[1], y1_4_max[1], y1_5_max[1], y1_6_max[1], y1_7_max[1]]
    # max_list = [ y0_0_max[ 0 ], y0_1_max[ 0 ], y0_2_max[ 0 ], y0_3_max[ 0 ],
    #              y1_0_max[ 0 ], y1_1_max[ 0 ], y1_2_max[ 0 ], y1_3_max[ 0 ]]
    # len_list = [ y0_0_max[ 1 ], y0_1_max[ 1 ], y0_2_max[ 1 ], y0_3_max[ 1 ],
    #              y1_0_max[ 1 ], y1_1_max[ 1 ], y1_2_max[ 1 ], y1_3_max[ 1 ]]

    # print(max_list)
    range_ = max(max_list)
    min_ = len_list[max_list.index(range_)]

    y0_0 = min_max(y0_0, range_, min(y0_0))
    y0_1 = min_max(y0_1, range_, min(y0_1))
    y0_2 = min_max(y0_2, range_, min(y0_2))
    y0_3 = min_max(y0_3, range_, min(y0_3))
    y0_4 = min_max(y0_4, range_, min(y0_4))
    y0_5 = min_max(y0_5, range_, min(y0_5))
    y0_6 = min_max(y0_6, range_, min(y0_6))
    y0_7 = min_max(y0_7, range_, min(y0_7))

    y1_0 = min_max(y1_0, range_, min(y1_0))
    y1_1 = min_max(y1_1, range_, min(y1_1))
    y1_2 = min_max(y1_2, range_, min(y1_2))
    y1_3 = min_max(y1_3, range_, min(y1_3))
    y1_4 = min_max(y1_4, range_, min(y1_4))
    y1_5 = min_max(y1_5, range_, min(y1_5))
    y1_6 = min_max(y1_6, range_, min(y1_6))
    y1_7 = min_max(y1_7, range_, min(y1_7))

    y0_0 = center2(y0_0)
    y0_1 = center2(y0_1)
    y0_2 = center2(y0_2)
    y0_3 = center2(y0_3)
    y0_4 = center2(y0_4)
    y0_5 = center2(y0_5)
    y0_6 = center2(y0_6)
    y0_7 = center2(y0_7)

    y1_0 = center2(y1_0)
    y1_1 = center2(y1_1)
    y1_2 = center2(y1_2)
    y1_3 = center2(y1_3)
    y1_4 = center2(y1_4)
    y1_5 = center2(y1_5)
    y1_6 = center2(y1_6)
    y1_7 = center2(y1_7)

    y0_0_max = max_ch(y0_0)
    y0_1_max = max_ch(y0_1)
    y0_2_max = max_ch(y0_2)
    y0_3_max = max_ch(y0_3)
    y0_4_max = max_ch(y0_4)
    y0_5_max = max_ch(y0_5)
    y0_6_max = max_ch(y0_6)
    y0_7_max = max_ch(y0_7)

    y1_0_max = max_ch(y1_0)
    y1_1_max = max_ch(y1_1)
    y1_2_max = max_ch(y1_2)
    y1_3_max = max_ch(y1_3)
    y1_4_max = max_ch(y1_4)
    y1_5_max = max_ch(y1_5)
    y1_6_max = max_ch(y1_6)
    y1_7_max = max_ch(y1_7)

    # max_list = [y0_0_max[0], y0_1_max[0], y0_2_max[0], y0_3_max[0], y0_4_max[0], y0_5_max[0], y0_6_max[0], #
    #             y1_0_max[0], y1_1_max[0], y1_2_max[0], y1_3_max[0], y1_4_max[0], y1_5_max[0], y1_6_max[0]] #
    # len_list = [y0_0_max[1], y0_1_max[1], y0_2_max[1], y0_3_max[1], y0_4_max[1], y0_5_max[1], y0_6_max[1], #
    #             y1_0_max[1], y1_1_max[1], y1_2_max[1], y1_3_max[1], y1_4_max[1], y1_5_max[1], y1_6_max[1]] #

    max_0 = [y0_0_max[0], y1_0_max[0]]
    max_1 = [y0_1_max[0], y1_1_max[0]]
    max_2 = [y0_2_max[0], y1_2_max[0]]
    max_3 = [y0_3_max[0], y1_3_max[0]]
    max_4 = [y0_4_max[0], y1_4_max[0]]
    max_5 = [y0_5_max[0], y1_5_max[0]]
    max_6 = [y0_6_max[0], y1_6_max[0]]
    max_7 = [y0_7_max[0], y1_7_max[0]]

    range_0 = max(max_0)
    range_1 = max(max_1)
    range_2 = max(max_2)
    range_3 = max(max_3)
    range_4 = max(max_4)
    range_5 = max(max_5)
    range_6 = max(max_6)
    range_7 = max(max_7)
    min_ = len_list[max_list.index(range_)]

    y0_0 = min_max(y0_0, range_0, min(y0_0))
    y0_1 = min_max(y0_1, range_1, min(y0_1))
    y0_2 = min_max(y0_2, range_2, min(y0_2))
    y0_3 = min_max(y0_3, range_3, min(y0_3))
    y0_4 = min_max(y0_4, range_4, min(y0_4))
    y0_5 = min_max(y0_5, range_5, min(y0_5))
    y0_6 = min_max(y0_6, range_6, min(y0_6))
    y0_7 = min_max(y0_7, range_7, min(y0_7))

    y1_0 = min_max(y1_0, range_0, min(y1_0))
    y1_1 = min_max(y1_1, range_1, min(y1_1))
    y1_2 = min_max(y1_2, range_2, min(y1_2))
    y1_3 = min_max(y1_3, range_3, min(y1_3))
    y1_4 = min_max(y1_4, range_4, min(y1_4))
    y1_5 = min_max(y1_5, range_5, min(y1_5))
    y1_6 = min_max(y1_6, range_6, min(y1_6))
    y1_7 = min_max(y1_7, range_7, min(y1_7))

    # y0_0 = center2(y0_0)
    # y0_1 = center2(y0_1)
    # y0_2 = center2(y0_2)
    # y0_3 = center2(y0_3)
    # y0_4 = center2(y0_4)
    # y0_5 = center2(y0_5)
    # y0_6 = center2(y0_6)
    #
    # y1_0 = center2(y1_0)
    # y1_1 = center2(y1_1)
    # y1_2 = center2(y1_2)
    # y1_3 = center2(y1_3)
    # y1_4 = center2(y1_4)
    # y1_5 = center2(y1_5)
    # y1_6 = center2(y1_6)

    new_ch_list_0 = [y0_0, y0_1, y0_2, y0_3, y0_4, y0_5, y0_6, y0_7]
    new_ch_list_1 = [y1_0, y1_1, y1_2, y1_3, y1_4, y1_5, y1_6, y1_7]
    # new_ch_list_0 = [ y0_0, y0_1, y0_2, y0_3]
    # new_ch_list_1 = [ y1_0, y1_1, y1_2, y1_3]

    new_ch_list = [new_ch_list_0, new_ch_list_1]

    return new_ch_list

def mapping_ssc(ch_list):
    len_num = len(ch_list[0])
    x = np.linspace(0, len_num, len_num)

    sscaler.fit(ch_list)
    new_ch_list = sscaler.transform(ch_list)

    # new_ch_list = [y0, y1, y2, y3, y4, y5, y6]

    return new_ch_list

def outlier_iqr(d):

    for i in range(len(d.columns)):
        # 四分位数
        a = d.iloc[:, i]

        Q1 = a.quantile(.25)
        Q3 = a.quantile(.75)
        # 四分位数
        q = Q3 - Q1

        # 外れ値の基準点
        outlier_min = Q1 - q * 1.5
        outlier_max = Q3 + q * 1.5

        # 範囲から外れている値を除く
        a[a < outlier_min] = outlier_min
        a[a > outlier_max] = outlier_max

    return d

def sim_pearson(d1, d2):
    n = len(d1)

    mean1 = sum(d1) / n
    mean2 = sum(d2) / n

    variance1 = 0
    variance2 = 0
    covariance = 0
    for k in range(n):
        a1 = (d1[k] - mean1)
        variance1 += a1 ** 2

        a2 = (d2[k] - mean2)
        variance2 += a2 ** 2

        covariance += a1 * a2

    variance1 = math.sqrt(variance1)
    variance2 = math.sqrt(variance2)

    if variance1 * variance2 == 0 : return 0

    return covariance / (variance1 * variance2)




def write_plot(ch_list, name):
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
    y7 = ch_list[7]

    labels0 = [-0.5, 0.5]
    labels1 = [-0.5, 0.5]
    labels2 = [-0.5, 0.5]
    labels3 = [-0.5, 0.5]
    labels4 = [-0.5, 0.5]
    labels5 = [-0.5, 0.5]
    labels6 = [-0.5, 0.5]
    labels7 = [-0.5, 0.5]

    l0, l1, l2, l3, l4, l5, l6, l7 = "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"
    o0, o1, o2, o3, o4, o5, o6, o7 = 1, 3, 5, 7, 9, 11, 13, 15
    # l0, l1, l2, l3 = "ch1", "ch2", "ch3", "ch4"
    # o0, o1, o2, o3 = 1, 3, 5, 7
    # o0 *= 5
    # o1 *= 5
    # o2 *= 5
    # o3 *= 5
    # o4 *= 5
    # o5 *= 5
    # o6 *= 5

    yticks0 = [la + o0 for la in labels0]
    yticks1 = [la + o1 for la in labels1]
    yticks2 = [la + o2 for la in labels2]
    yticks3 = [la + o3 for la in labels3]
    yticks4 = [la + o4 for la in labels4]
    yticks5 = [la + o5 for la in labels5]
    yticks6 = [la + o6 for la in labels6]
    yticks7 = [la + o7 for la in labels7]

    ytls = labels0 + labels1 + labels2 + labels3 + labels4 + labels5 + labels6 + labels7
    ytks = yticks0 + yticks1 + yticks2 + yticks3 + yticks4 + yticks5 + yticks6 + yticks7
    plt.figure(figsize=(6, 5), facecolor="w")
    yo0 = list(map(lambda x: x + o7, y0))
    yo1 = list(map(lambda x: x + o6, y1))
    yo2 = list(map(lambda x: x + o5, y2))
    yo3 = list(map(lambda x: x + o4, y3))
    yo4 = list(map(lambda x: x + o3, y4))
    yo5 = list(map(lambda x: x + o2, y5))
    yo6 = list(map(lambda x: x + o1, y6))
    yo7 = list(map(lambda x: x + o0, y7))

    plt.plot(x, yo0, color="b", label=l0)
    plt.plot(x, yo1, color="r", label=l1)
    plt.plot(x, yo2, color="g", label=l2)
    plt.plot(x, yo3, color="c", label=l3)
    plt.plot(x, yo4, color="m", label=l4)
    plt.plot(x, yo5, color="y", label=l5)
    plt.plot(x, yo6, color="k", label=l6)
    plt.plot(x, yo7, color='#f781bf', label=l7)

    # ytls = labels0 + labels1 + labels2 + labels3
    # ytks = yticks0 + yticks1 + yticks2 + yticks3
    # plt.figure(figsize=(6, 5), facecolor="w")
    # yo0 = list(map(lambda x: x + o3, y0))
    # yo1 = list(map(lambda x: x + o2, y1))
    # yo2 = list(map(lambda x: x + o1, y2))
    # yo3 = list(map(lambda x: x + o0, y3))
    #
    # plt.plot(x, yo0, color="b", label=l0)
    # plt.plot(x, yo1, color="r", label=l1)
    # plt.plot(x, yo2, color="g", label=l2)
    # plt.plot(x, yo3, color="c", label=l3)

    plt.ylim(o0 - 2, o7 + 2)
    plt.yticks(ytks)
    plt.axes().set_yticklabels(ytls)
    plt.legend(loc="upper right", fontsize=8)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(labelright=False, labeltop=False)
    plt.tick_params(right=False, top=False)
    plt.title(name)

    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if plot_flg == 1:
        if x_flg == 1:
            # plotpath2 = path.png_ica + "/" + d + "/" + i + "/複合+複合2/" + name
            plotpath2 = path.png_ica_new + "/" + i + "/" + d + "/" + s + "/" + "fastICA" + "/" + name
        elif x_flg == 2:
            plotpath2 = path.png_ica + "/" + d + "/" + i + "/複合+手首/" + name
        else:
            plotpath2 = path.png_ica + "/" + d + "/" + i + "/複合+指/" + name
        plt.savefig(plotpath2)
        # plt.show()
    else: plt.show()
    plt.close('all')





nf_all = open(path.png_ica_new + "/" + "結果まとめ_fastICA.CSV", 'w', encoding="utf_8_sig")
dataWriter_all = csv.writer(nf_all)
all_average = 0
for i in path.new_subject:
    sub_average = 0
    dataWriter_all.writerow([])
    dataWriter_all.writerow([i])
    for d in path.new_day:
        dataWriter_all.writerow([d])
        for s in path.sets:
            dataWriter_all.writerow([s, "1ch", "2ch", "3ch", "4ch", "5ch", "6ch", "7ch", "8ch", "平均"])
            # 複合動作
            for j in path.mix_only:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + l +"fft.CSV"
                    elif fft_flg == 2:
                        file_path = path.plus_avarage_fft_long + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
                    else:
                        file_path = path.plus_avarage_ifft + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                    mix1 = pd.read_csv(file_path)
                    # mix1 = outlier_iqr(mix1)

                    mix_ch0 = mix1.iloc[:, 0].values
                    mix_ch1 = mix1.iloc[:, 1].values
                    mix_ch2 = mix1.iloc[:, 2].values
                    mix_ch3 = mix1.iloc[:, 3].values
                    mix_ch4 = mix1.iloc[:, 4].values
                    mix_ch5 = mix1.iloc[:, 5].values
                    mix_ch6 = mix1.iloc[:, 6].values
                    mix_ch7 = mix1.iloc[:, 7].values

                    print("複合1：/" + i + "/" + d + "/" + s + "/" + j + "/" + l + "fft.CSV")


            for j in path.mix_only2:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + l +"fft.CSV"
                    elif fft_flg == 2:
                        file_path = path.plus_avarage_fft_long + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
                    else:
                        file_path = path.plus_avarage_ifft + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                    mix2 = pd.read_csv(file_path)
                    # mix2 = outlier_iqr(mix2)

                    mix_2_ch0 = mix2.iloc[:, 0].values
                    mix_2_ch1 = mix2.iloc[:, 1].values
                    mix_2_ch2 = mix2.iloc[:, 2].values
                    mix_2_ch3 = mix2.iloc[:, 3].values
                    mix_2_ch4 = mix2.iloc[:, 4].values
                    mix_2_ch5 = mix2.iloc[:, 5].values
                    mix_2_ch6 = mix2.iloc[:, 6].values
                    mix_2_ch7 = mix2.iloc[:, 7].values

                    print("複合2：/" + i + "/" + d + "/" + s + "/" + j + "/" + l + "fft.CSV")

            # 手首動作
            for j in path.tekubi_only:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + l +"fft.CSV"
                    elif fft_flg == 2:
                        file_path = path.plus_avarage_fft_long + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
                    else:
                        file_path = path.plus_avarage_ifft + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                    tekubi = pd.read_csv(file_path)
                    # tekubi = outlier_iqr(tekubi)

                    tekubi_ch0 = tekubi.iloc[:, 0].values
                    tekubi_ch1 = tekubi.iloc[:, 1].values
                    tekubi_ch2 = tekubi.iloc[:, 2].values
                    tekubi_ch3 = tekubi.iloc[:, 3].values
                    tekubi_ch4 = tekubi.iloc[:, 4].values
                    tekubi_ch5 = tekubi.iloc[:, 5].values
                    tekubi_ch6 = tekubi.iloc[:, 6].values
                    tekubi_ch7 = tekubi.iloc[:, 7].values

                    print("手首：/" + i + "/" + d + "/" + s + "/" + j + "/" + l + "fft.CSV")

            # 指動作
            for j in path.yubi_only:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + l +"fft.CSV"
                    elif fft_flg == 2:
                        file_path = path.plus_avarage_fft_long + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
                    else:
                        file_path = path.plus_avarage_ifft + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                    yubi = pd.read_csv(file_path)
                    # yubi = outlier_iqr(yubi)

                    yubi_ch0 = yubi.iloc[:, 0].values
                    yubi_ch1 = yubi.iloc[:, 1].values
                    yubi_ch2 = yubi.iloc[:, 2].values
                    yubi_ch3 = yubi.iloc[:, 3].values
                    yubi_ch4 = yubi.iloc[:, 4].values
                    yubi_ch5 = yubi.iloc[:, 5].values
                    yubi_ch6 = yubi.iloc[:, 6].values
                    yubi_ch7 = yubi.iloc[:, 7].values

                    print("指：/" + i + "/" + d + "/" + s + "/" + j + "/" + l + "fft.CSV")


            X0 = mix_sources([mix_ch0, mix_2_ch0])
            X1 = mix_sources([mix_ch1, mix_2_ch1])
            X2 = mix_sources([mix_ch2, mix_2_ch2])
            X3 = mix_sources([mix_ch3, mix_2_ch3])
            X4 = mix_sources([mix_ch4, mix_2_ch4])
            X5 = mix_sources([mix_ch5, mix_2_ch5])
            X6 = mix_sources([mix_ch6, mix_2_ch6])
            X7 = mix_sources([mix_ch7, mix_2_ch7])

            # elif x_flg == 2:
            #     X0 = mix_sources([mix_ch0, tekubi_ch0])
            #     X1 = mix_sources([mix_ch1, tekubi_ch1])
            #     X2 = mix_sources([mix_ch2, tekubi_ch2])
            #     X3 = mix_sources([mix_ch3, tekubi_ch3])
            #     # X4 = mix_sources([mix_ch4, tekubi_ch4]) #
            #     # X5 = mix_sources([mix_ch5, tekubi_ch5]) #
            #     # X6 = mix_sources([mix_ch6, tekubi_ch6]) #
            #
            # else:
            #     X0 = mix_sources([ mix_ch0, yubi_ch0 ])
            #     X1 = mix_sources([ mix_ch1, yubi_ch1 ])
            #     X2 = mix_sources([ mix_ch2, yubi_ch2 ])
            #     X3 = mix_sources([ mix_ch3, yubi_ch3 ])
            #     # X4 = mix_sources([ mix_ch4, yubi_ch4 ]) #
            #     # X5 = mix_sources([ mix_ch5, yubi_ch5 ]) #
            #     # X6 = mix_sources([ mix_ch6, yubi_ch6 ]) #

            # X0 = mix_sources([tekubi_ch0, yubi_ch0])
            # X1 = mix_sources([tekubi_ch1, yubi_ch1])
            # X2 = mix_sources([tekubi_ch2, yubi_ch2])
            # X3 = mix_sources([tekubi_ch3, yubi_ch3])
            # X4 = mix_sources([tekubi_ch4, yubi_ch4])
            # X5 = mix_sources([tekubi_ch5, yubi_ch5])
            # X6 = mix_sources([tekubi_ch6, yubi_ch6])

            # S0 = ica(X0, iterations=1000)
            # S1 = ica(X1, iterations=1000)
            # S2 = ica(X2, iterations=1000)
            # S3 = ica(X3, iterations=1000)
            # S4 = ica(X4, iterations=1000)
            # S5 = ica(X5, iterations=1000)
            # S6 = ica(X6, iterations=1000)
            # S7 = ica(X7, iterations=1000)
            # print(S0)

            S0 = SICA.fit_transform(X0)
            S1 = SICA.fit_transform(X1)
            S2 = SICA.fit_transform(X2)
            S3 = SICA.fit_transform(X3)
            S4 = SICA.fit_transform(X4)
            S5 = SICA.fit_transform(X5)
            S6 = SICA.fit_transform(X6)
            S7 = SICA.fit_transform(X7)

            # print(S0)


            # S0 = np.array(jadeR(X0))
            # S1 = np.array(jadeR(X1))
            # S2 = np.array(jadeR(X2))
            # S3 = np.array(jadeR(X3))

            # S0 = ica1(X0, 2)
            # S1 = ica1(X1, 2)
            # S2 = ica1(X2, 2)
            # S3 = ica1(X3, 2)

            # print("S")
            # print(S0.ndim)
            #
            # print("S0[0]")
            # print(S0[0].ndim)


            tekubi_ch_list = [tekubi_ch0, tekubi_ch1, tekubi_ch2, tekubi_ch3, tekubi_ch4, tekubi_ch5, tekubi_ch6, tekubi_ch7]
            mix_ch_list = [mix_ch0, mix_ch1, mix_ch2, mix_ch3, mix_ch4, mix_ch5, mix_ch6, mix_ch7]
            # mix_2_ch_list = [mix_2_ch0, mix_2_ch1, mix_2_ch2, mix_2_ch3, mix_2_ch4, mix_2_ch5, mix_2_ch6] #
            yubi_ch_list = [yubi_ch0, yubi_ch1, yubi_ch2, yubi_ch3, yubi_ch4, yubi_ch5, yubi_ch6, yubi_ch7]
            S_0_ch_list = [S0[0], S1[0], S2[0], S3[0], S4[0], S5[0], S6[0], S7[0]]
            S_1_ch_list = [S0[1], S1[1], S2[1], S3[1], S4[1], S5[1], S6[1], S7[1]]
            X_0_ch_list = [X0[0], X1[0], X2[0], X3[0], X4[0], X5[0], X6[0], X7[0]]
            X_1_ch_list = [X0[1], X1[1], X2[1], X3[1], X4[1], X5[1], X6[1], X7[1]]
            S_ch_list = [S0, S1, S2, S3, S4, S5, S6, S7]
            X_ch_list = [X0, X1, X2, X3, X4, X5, X6, X7]


            tekubi_ch_list_abs = [abs(tekubi_ch0), abs(tekubi_ch1), abs(tekubi_ch2), abs(tekubi_ch3),
                                  abs(tekubi_ch4), abs(tekubi_ch5), abs(tekubi_ch6), abs(tekubi_ch7)]
            yubi_ch_list_abs = [abs(yubi_ch0), abs(yubi_ch1), abs(yubi_ch2), abs(yubi_ch3),
                                abs(yubi_ch4), abs(yubi_ch5), abs(yubi_ch6), abs(yubi_ch7)]
            mix_ch_list_abs = [abs(mix_ch0), abs(mix_ch1), abs(mix_ch2), abs(mix_ch3),
                               abs(mix_ch4), abs(mix_ch5), abs(mix_ch6), abs(mix_ch7)]
            mix_2_ch_list_abs = [abs(mix_2_ch0), abs(mix_2_ch1), abs(mix_2_ch2), abs(mix_2_ch3), #
                               abs(mix_2_ch4), abs(mix_2_ch5), abs(mix_2_ch6)] #
            S_ch_list_abs = [abs(S0), abs(S1), abs(S2), abs(S3), abs(S4), abs(S5), abs(S6), abs(S7)]
            X_ch_list_abs = [abs(X0), abs(X1), abs(X2), abs(X3), abs(X4), abs(X5), abs(X6), abs(X7)]
            S_0_ch_list_abs = [abs(S0[0]), abs(S1[0]), abs(S2[0]), abs(S3[0]), abs(S4[0]),
                               abs(S5[0]), abs(S6[0]), abs(S7[0])]
            S_1_ch_list_abs = [abs(S0[1]), abs(S1[1]), abs(S2[1]), abs(S3[1]), abs(S4[1]),
                               abs(S5[1]), abs(S6[1]), abs(S7[1])]
            X_0_ch_list_abs = [abs(X0[0]), abs(X1[0]), abs(X2[0]), abs(X3[0]), abs(X4[0]),
                               abs(X5[0]), abs(X6[0]), abs(X7[0])]
            X_1_ch_list_abs = [abs(X0[1]), abs(X1[1]), abs(X2[1]), abs(X3[1]), abs(X4[1]),
                               abs(X5[1]), abs(X6[1]), abs(X7[1])]

            # tekubi_ch_list = [tekubi_ch0, tekubi_ch1, tekubi_ch2, tekubi_ch3]
            # mix_ch_list = [mix_ch0, mix_ch1, mix_ch2, mix_ch3]
            # # mix_2_ch_list = [mix_2_ch0, mix_2_ch1, mix_2_ch2, mix_2_ch3, mix_2_ch4, mix_2_ch5, mix_2_ch6]
            # yubi_ch_list = [yubi_ch0, yubi_ch1, yubi_ch2, yubi_ch3]
            # S_0_ch_list = [S0[0], S1[0], S2[0], S3[0]]
            # S_1_ch_list = [S0[1], S1[1], S2[1], S3[1]]
            # X_0_ch_list = [X0[0], X1[0], X2[0], X3[0]]
            # X_1_ch_list = [X0[1], X1[1], X2[1], X3[1]]
            # S_ch_list = [S0, S1, S2, S3]
            # X_ch_list = [X0, X1, X2, X3]
            #
            # tekubi_ch_list_abs = [abs(tekubi_ch0)**2, abs(tekubi_ch1)**2, abs(tekubi_ch2)**2, abs(tekubi_ch3)**2]
            # yubi_ch_list_abs = [abs(yubi_ch0)**2, abs(yubi_ch1)**2, abs(yubi_ch2)**2, abs(yubi_ch3)**2]
            # mix_ch_list_abs = [abs(mix_ch0)**2, abs(mix_ch1)**2, abs(mix_ch2)**2, abs(mix_ch3)**2]
            # # mix_2_ch_list_abs = [abs(mix_2_ch0)**2, abs(mix_2_ch1)**2, abs(mix_2_ch2)**2, abs(mix_2_ch3)**2,
            # #                    abs(mix_2_ch4)**2, abs(mix_2_ch5)**2, abs(mix_2_ch6)**2]
            # S_ch_list_abs = [abs(S0)**2, abs(S1)**2, abs(S2)**2, abs(S3)**2]
            # X_ch_list_abs = [abs(X0)**2, abs(X1)**2, abs(X2)**2, abs(X3)**2]
            # S_0_ch_list_abs = [abs(S0[0])**2, abs(S1[0])**2, abs(S2[0])**2, abs(S3[0])**2]
            # S_1_ch_list_abs = [abs(S0[1])**2, abs(S1[1])**2, abs(S2[1])**2, abs(S3[1])**2]
            # X_0_ch_list_abs = [abs(X0[0])**2, abs(X1[0])**2, abs(X2[0])**2, abs(X3[0])**2]
            # X_1_ch_list_abs = [abs(X0[1])**2, abs(X1[1])**2, abs(X2[1])**2, abs(X3[1])**2]

            if fft_flg == 0:
                origin_list_m = mapping_1(yubi_ch_list, tekubi_ch_list)
                S_list = mapping_1(S_0_ch_list, S_1_ch_list)
                mix_list = mapping_1(X_0_ch_list, X_1_ch_list)
                # mix_list = mapping_1(mix_ch_list, mix_2_ch_list)

            else:
                origin_list_m = mapping_1(yubi_ch_list_abs, tekubi_ch_list_abs)
                S_list = mapping_1(S_0_ch_list_abs, S_1_ch_list_abs)
                mix_list = mapping_1(X_0_ch_list_abs, X_1_ch_list_abs)
                # mix_list = mapping_1(mix_ch_list_abs, mix_2_ch_list_abs)
                # mix_list_abs = mapping_1(mix_ch_list, mix_2_ch_list)


            yubi_ch_list_m = origin_list_m[0]
            tekubi_ch_list_m = origin_list_m[1]
            S_0_ch_list_m = S_list[0]
            S_1_ch_list_m = S_list[1]
            mix_ch_list_m = mix_list[0]
            mix_2_ch_list_m = mix_list[1]
            # mix_ch_list_m = mix_list[0]
            # mix_2_ch_list_m = mix_list[1]


            y_0p = pd.DataFrame(yubi_ch_list_m[0])
            y_0v = y_0p.var().values
            y_1p = pd.DataFrame(yubi_ch_list_m[1])
            y_1v = y_1p.var().values
            y_2p = pd.DataFrame(yubi_ch_list_m[2])
            y_2v = y_2p.var().values
            y_3p = pd.DataFrame(yubi_ch_list_m[3])
            y_3v = y_3p.var().values
            y_4p = pd.DataFrame(yubi_ch_list_m[4])
            y_4v = y_4p.var().values
            y_5p = pd.DataFrame(yubi_ch_list_m[5])
            y_5v = y_5p.var().values
            y_6p = pd.DataFrame(yubi_ch_list_m[6])
            y_6v = y_6p.var().values
            y_7p = pd.DataFrame(yubi_ch_list_m[7])
            y_7v = y_7p.var().values

            yubi_var = [y_0v, y_1v, y_2v, y_3v, y_4v, y_5v, y_6v, y_7v]
            # yubi_var = [y_0v, y_1v, y_2v, y_3v]
            # print(yubi_var)

            t_0p = pd.DataFrame(tekubi_ch_list_m[0])
            t_0v = t_0p.var().values
            t_1p = pd.DataFrame(tekubi_ch_list_m[1])
            t_1v = t_1p.var().values
            t_2p = pd.DataFrame(tekubi_ch_list_m[2])
            t_2v = t_2p.var().values
            t_3p = pd.DataFrame(tekubi_ch_list_m[3])
            t_3v = t_3p.var().values
            t_4p = pd.DataFrame(tekubi_ch_list_m[4])
            t_4v = t_4p.var().values
            t_5p = pd.DataFrame(tekubi_ch_list_m[5])
            t_5v = t_5p.var().values
            t_6p = pd.DataFrame(tekubi_ch_list_m[6])
            t_6v = t_6p.var().values
            t_7p = pd.DataFrame(tekubi_ch_list_m[7])
            t_7v = t_7p.var().values

            tekubi_var = [t_0v, t_1v, t_2v, t_3v, t_4v, t_5v, t_6v, t_7v]
            # tekubi_var = [t_0v, t_1v, t_2v, t_3v]
            # print(tekubi_ver)

            S0_0p = pd.DataFrame(S_0_ch_list_m[0])
            S0_0v = S0_0p.var().values
            S0_1p = pd.DataFrame(S_0_ch_list_m[1])
            S0_1v = S0_1p.var().values
            S0_2p = pd.DataFrame(S_0_ch_list_m[2])
            S0_2v = S0_2p.var().values
            S0_3p = pd.DataFrame(S_0_ch_list_m[3])
            S0_3v = S0_3p.var().values
            S0_4p = pd.DataFrame(S_0_ch_list_m[4])
            S0_4v = S0_4p.var().values
            S0_5p = pd.DataFrame(S_0_ch_list_m[5])
            S0_5v = S0_5p.var().values
            S0_6p = pd.DataFrame(S_0_ch_list_m[6])
            S0_6v = S0_6p.var().values
            S0_7p = pd.DataFrame(S_0_ch_list_m[7])
            S0_7v = S0_7p.var().values

            S0_var = [S0_0v, S0_1v, S0_2v, S0_3v, S0_4v, S0_5v, S0_6v, S0_7v]
            # S0_var = [S0_0v, S0_1v, S0_2v, S0_3v]
            # print(S0_ver)

            S1_0p = pd.DataFrame(S_1_ch_list_m[0])
            S1_0v = S1_0p.var().values
            S1_1p = pd.DataFrame(S_1_ch_list_m[1])
            S1_1v = S1_1p.var().values
            S1_2p = pd.DataFrame(S_1_ch_list_m[2])
            S1_2v = S1_2p.var().values
            S1_3p = pd.DataFrame(S_1_ch_list_m[3])
            S1_3v = S1_3p.var().values
            S1_4p = pd.DataFrame(S_1_ch_list_m[4])
            S1_4v = S1_4p.var().values
            S1_5p = pd.DataFrame(S_1_ch_list_m[5])
            S1_5v = S1_5p.var().values
            S1_6p = pd.DataFrame(S_1_ch_list_m[6])
            S1_6v = S1_6p.var().values
            S1_7p = pd.DataFrame(S_1_ch_list_m[7])
            S1_7v = S1_7p.var().values

            S1_var = [S1_0v, S1_1v, S1_2v, S1_3v, S1_4v, S1_5v, S1_6v, S1_7v]
            # S1_var = [S1_0v, S1_1v, S1_2v, S1_3v]
            # print(S0_ver)

            Mix_0p = pd.DataFrame(mix_ch_list_m[0])
            Mix_0v = Mix_0p.var().values
            Mix_1p = pd.DataFrame(mix_ch_list_m[1])
            Mix_1v = Mix_1p.var().values
            Mix_2p = pd.DataFrame(mix_ch_list_m[2])
            Mix_2v = Mix_2p.var().values
            Mix_3p = pd.DataFrame(mix_ch_list_m[3])
            Mix_3v = Mix_3p.var().values
            Mix_4p = pd.DataFrame(mix_ch_list_m[4])
            Mix_4v = Mix_4p.var().values
            Mix_5p = pd.DataFrame(mix_ch_list_m[5])
            Mix_5v = Mix_5p.var().values
            Mix_6p = pd.DataFrame(mix_ch_list_m[6])
            Mix_6v = Mix_6p.var().values
            Mix_7p = pd.DataFrame(mix_ch_list_m[7])
            Mix_7v = Mix_7p.var().values

            Mix_var = [Mix_0v, Mix_1v, Mix_2v, Mix_3v, Mix_4v, Mix_5v, Mix_6v, Mix_7v]
            # Mix_var = [Mix_0v, Mix_1v, Mix_2v, Mix_3v]
            # print(S0_ver)


            S_0 = []
            S_0.append(S_0_ch_list_m[0])
            S_0.append(S_0_ch_list_m[1])
            S_0.append(S_0_ch_list_m[2])
            S_0.append(S_0_ch_list_m[3])
            S_0.append(S_0_ch_list_m[4])
            S_0.append(S_0_ch_list_m[5])
            S_0.append(S_0_ch_list_m[6])
            S_0.append(S_0_ch_list_m[7])

            S_0p = pd.DataFrame(S_0)
            S_0 = S_0p.T
            # print(S_0.shape)

            S_1 = []
            S_1.append(S_1_ch_list_m[0])
            S_1.append(S_1_ch_list_m[1])
            S_1.append(S_1_ch_list_m[2])
            S_1.append(S_1_ch_list_m[3])
            S_1.append(S_1_ch_list_m[4])
            S_1.append(S_1_ch_list_m[5])
            S_1.append(S_1_ch_list_m[6])
            S_1.append(S_1_ch_list_m[7])

            S_1p = pd.DataFrame(S_1)
            S_1 = S_1p.T
            # print(S_1.shape)

            S_ch_0 = []
            S_ch_0.append(S_0_ch_list_m[0])
            S_ch_0.append(S_1_ch_list_m[0])
            S_ch_0 = pd.DataFrame(S_ch_0)
            S_ch_0 = S_ch_0.T
            # print(S_ch_0.shape)

            S_ch_1 = []
            S_ch_1.append(S_0_ch_list_m[1])
            S_ch_1.append(S_1_ch_list_m[1])
            S_ch_1 = pd.DataFrame(S_ch_1)
            S_ch_1 = S_ch_1.T
            # print(S_ch_1.shape)

            S_ch_2 = []
            S_ch_2.append(S_0_ch_list_m[2])
            S_ch_2.append(S_1_ch_list_m[2])
            S_ch_2 = pd.DataFrame(S_ch_2)
            S_ch_2 = S_ch_2.T
            # print(S_ch_2.shape)

            S_ch_3 = []
            S_ch_3.append(S_0_ch_list_m[3])
            S_ch_3.append(S_1_ch_list_m[3])
            S_ch_3 = pd.DataFrame(S_ch_3)
            S_ch_3 = S_ch_3.T
            # print(S_ch_3.shape)

            S_ch_4 = []
            S_ch_4.append(S_0_ch_list_m[4])
            S_ch_4.append(S_1_ch_list_m[4])
            S_ch_4 = pd.DataFrame(S_ch_4)
            S_ch_4 = S_ch_4.T
            # print(S_ch_4.shape) #

            S_ch_5 = []
            S_ch_5.append(S_0_ch_list_m[5])
            S_ch_5.append(S_1_ch_list_m[5])
            S_ch_5 = pd.DataFrame(S_ch_5)
            S_ch_5 = S_ch_5.T
            # print(S_ch_5.shape) #

            S_ch_6 = []
            S_ch_6.append(S_0_ch_list_m[6])
            S_ch_6.append(S_1_ch_list_m[6])
            S_ch_6 = pd.DataFrame(S_ch_6)
            S_ch_6 = S_ch_6.T
            # print(S_ch_6.shape) #

            S_ch_7 = []
            S_ch_7.append(S_0_ch_list_m[7])
            S_ch_7.append(S_1_ch_list_m[7])
            S_ch_7 = pd.DataFrame(S_ch_7)
            S_ch_7 = S_ch_7.T
            # print(S_ch_6.shape) #

            tekubi = []
            tekubi.append(tekubi_ch_list_m[0])
            tekubi.append(tekubi_ch_list_m[1])
            tekubi.append(tekubi_ch_list_m[2])
            tekubi.append(tekubi_ch_list_m[3])
            tekubi.append(tekubi_ch_list_m[4])
            tekubi.append(tekubi_ch_list_m[5])
            tekubi.append(tekubi_ch_list_m[6])
            tekubi.append(tekubi_ch_list_m[7])

            tekubi_p = pd.DataFrame(tekubi)
            tekubi = tekubi_p.T

            # print(tekubi)


            yubi = []
            yubi.append(yubi_ch_list_m[0])
            yubi.append(yubi_ch_list_m[1])
            yubi.append(yubi_ch_list_m[2])
            yubi.append(yubi_ch_list_m[3])
            yubi.append(yubi_ch_list_m[4])
            yubi.append(yubi_ch_list_m[5])
            yubi.append(yubi_ch_list_m[6])
            yubi.append(yubi_ch_list_m[7])

            yubi_p = pd.DataFrame(yubi)
            yubi = yubi_p.T
            # print(yubi.shape)

            X_0 = []
            X_0.append(mix_ch_list_m[0])
            X_0.append(mix_ch_list_m[1])
            X_0.append(mix_ch_list_m[2])
            X_0.append(mix_ch_list_m[3])
            X_0.append(mix_ch_list_m[4])
            X_0.append(mix_ch_list_m[5])
            X_0.append(mix_ch_list_m[6])
            X_0.append(mix_ch_list_m[7])

            X_0_p = pd.DataFrame(X_0)
            X_0 = X_0_p.T
            # print(X_0.shape)

            X_1 = []
            X_1.append(mix_2_ch_list_m[0])
            X_1.append(mix_2_ch_list_m[1])
            X_1.append(mix_2_ch_list_m[2])
            X_1.append(mix_2_ch_list_m[3])
            X_1.append(mix_2_ch_list_m[4])
            X_1.append(mix_2_ch_list_m[5])
            X_1.append(mix_2_ch_list_m[6])
            X_1.append(mix_2_ch_list_m[7])

            X_1_p = pd.DataFrame(X_1)
            X_1 = X_1_p.T
            # print(X_1.shape)

            mix_ = []
            mix_.append(mix_ch_list_m[0])
            mix_.append(mix_ch_list_m[1])
            mix_.append(mix_ch_list_m[2])
            mix_.append(mix_ch_list_m[3])
            mix_.append(mix_ch_list_m[4])
            mix_.append(mix_ch_list_m[5])
            mix_.append(mix_ch_list_m[6])
            mix_.append(mix_ch_list_m[7])

            mix_p = pd.DataFrame(mix_)
            mix_ = mix_p.T


            # print(mix_.shape)

            # S_ch_0_0 = S_ch_0[0].values
            # S_ch_0_0 = S_ch_0_0.reshape((-1, 1))
            # S_ch_0_1 = S_ch_0[1].values
            # S_ch_0_1 = S_ch_0_1.reshape((-1, 1))
            #
            # S_ch_1_0 = S_ch_1[0].values
            # S_ch_1_0 = S_ch_1_0.reshape((-1, 1))
            # S_ch_1_1 = S_ch_1[1].values
            # S_ch_1_1 = S_ch_1_1.reshape((-1, 1))
            #
            # S_ch_2_0 = S_ch_2[0].values
            # S_ch_2_0 = S_ch_2_0.reshape((-1, 1))
            # S_ch_2_1 = S_ch_2[1].values
            # S_ch_2_1 = S_ch_2_1.reshape((-1, 1))
            #
            # S_ch_3_0 = S_ch_3[0].values
            # S_ch_3_0 = S_ch_3_0.reshape((-1, 1))
            # S_ch_3_1 = S_ch_3[1].values
            # S_ch_3_1 = S_ch_3_1.reshape((-1, 1))
            #
            # S_ch_4_0 = S_ch_4[0].values
            # S_ch_4_0 = S_ch_4_0.reshape((-1, 1))
            # S_ch_4_1 = S_ch_4[1].values
            # S_ch_4_1 = S_ch_4_1.reshape((-1, 1))
            #
            # S_ch_5_0 = S_ch_5[0].values
            # S_ch_5_0 = S_ch_5_0.reshape((-1, 1))
            # S_ch_5_1 = S_ch_5[1].values
            # S_ch_5_1 = S_ch_5_1.reshape((-1, 1))
            #
            # S_ch_6_0 = S_ch_6[0].values
            # S_ch_6_0 = S_ch_6_0.reshape((-1, 1))
            # S_ch_6_1 = S_ch_6[1].values
            # S_ch_6_1 = S_ch_6_1.reshape((-1, 1))

            print()
            print("////////////////////////////////////////////////////////")
            print("ch1")
            print()
            print(tekubi[0].shape)
            clf0.fit(S_ch_0, tekubi[0])
            print(clf0.coef_)
            b0_0 = (clf0.coef_[0] * ((S0_0v / t_0v) ** 0.5)) ** 2
            b0_1 = (clf0.coef_[1] * ((S1_0v / t_0v) ** 0.5)) ** 2
            ren0 = sim_pearson(S_ch_0[0], tekubi[0])
            ren0_1 = sim_pearson(S_ch_0[1], tekubi[0])
            score0 = clf0.score(S_ch_0, tekubi[0])

            clf0.fit(S_ch_0, yubi[0])
            print(clf0.coef_)
            b10_0 = (clf0.coef_[0] * ((S0_0v / y_0v) ** 0.5)) ** 2
            b10_1 = (clf0.coef_[1] * ((S1_0v / y_0v) ** 0.5)) ** 2
            ren10 = sim_pearson(S_ch_0[0], yubi[0])
            ren10_1 = sim_pearson(S_ch_0[1], yubi[0])
            score10 = clf0.score(S_ch_0, yubi[0])
            if abs(ren0) > abs(ren10):
                s_t0 = S_ch_0[0]
                s_y0 = S_ch_0[1]
            else:
                s_t0 = S_ch_0[1]
                s_y0 = S_ch_0[0]

            ren_list0 = [ren0, ren0_1, ren10, ren10_1]
            if ren_list0.index(max(ren_list0)) == 0 or ren_list0.index(max(ren_list0)) == 3:
                ren_list0 = [ ren_list0[ r ] for r in range(len(ren_list0)) if r == 0 or r == 3 ]
            else:
                ren_list0 = [ ren_list0[ r ] for r in range(len(ren_list0)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b0_0)
            print("標準化偏差回帰係数2")
            print(b0_1)
            print("S0の寄与率")
            print((b0_0 / (b0_0 + b0_1)) * 100)
            print("S1の寄与率")
            print((b0_1 / (b0_0 + b0_1)) * 100)
            print("相関係数")
            print(ren0)
            print(ren0_1)
            print("決定係数")
            print(score0)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b10_0)
            print("標準化偏差回帰係数2")
            print(b10_1)
            print("S0の寄与率")
            print((b10_0 / (b10_0 + b10_1)) * 100)
            print("S1の寄与率")
            print((b10_1 / (b10_0 + b10_1)) * 100)
            print("相関係数")
            print(ren10)
            print(ren10_1)
            print("決定係数")
            print(score10)

            print()
            print("////////////////////////////////////////////////////////")
            print("ch2")
            print()
            clf1.fit(S_ch_1, tekubi[1])
            print(clf1.coef_)
            b1_0 = (clf1.coef_[0] * ((S0_1v / t_1v) ** 0.5)) ** 2
            b1_1 = (clf1.coef_[1] * ((S1_1v / t_1v) ** 0.5)) ** 2
            ren1 = sim_pearson(S_ch_1[0], tekubi[1])
            ren1_1 = sim_pearson(S_ch_1[1], tekubi[1])
            score1 = clf1.score(S_ch_1, tekubi[1])

            clf1.fit(S_ch_1, yubi[1])
            print(clf1.coef_)
            b11_0 = (clf1.coef_[0] * ((S0_1v / y_1v) ** 0.5)) ** 2
            b11_1 = (clf1.coef_[1] * ((S1_1v / y_1v) ** 0.5)) ** 2
            ren11 = sim_pearson(S_ch_1[0], yubi[1])
            ren11_1 = sim_pearson(S_ch_1[1], yubi[1])
            score11 = clf1.score(S_ch_1, yubi[1])
            if abs(ren1) > abs(ren11):
                s_t1 = S_ch_1[0]
                s_y1 = S_ch_1[1]
            else:
                s_t1 = S_ch_1[1]
                s_y1 = S_ch_1[0]

            ren_list1 = [ ren1, ren1_1, ren11, ren11_1 ]
            if ren_list1.index(max(ren_list1)) == 0 or ren_list1.index(max(ren_list1)) == 3:
                ren_list1 = [ ren_list1[ r ] for r in range(len(ren_list1)) if r == 0 or r == 3 ]
            else:
                ren_list1 = [ ren_list1[ r ] for r in range(len(ren_list1)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b1_0)
            print("標準化偏差回帰係数2")
            print(b1_1)
            print("S0の寄与率")
            print((b1_0 / (b1_0 + b1_1)) * 100)
            print("S1の寄与率")
            print((b1_1 / (b1_0 + b1_1)) * 100)
            print("相関係数")
            print(ren1)
            print(ren1_1)
            print("決定係数")
            print(score1)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b11_0)
            print("標準化偏差回帰係数2")
            print(b11_1)
            print("S0の寄与率")
            print((b11_0 / (b11_0 + b11_1)) * 100)
            print("S1の寄与率")
            print((b11_1 / (b11_0 + b11_1)) * 100)
            print("相関係数")
            print(ren11)
            print(ren11_1)
            print("決定係数")
            print(score11)

            print()
            print("////////////////////////////////////////////////////////")
            print("ch3")
            print()
            clf2.fit(S_ch_2, tekubi[2])
            print(clf2.coef_)
            b2_0 = (clf2.coef_[0] * ((S0_2v / t_2v) ** 0.5)) ** 2
            b2_1 = (clf2.coef_[1] * ((S1_2v / t_2v) ** 0.5)) ** 2
            ren2 = sim_pearson(S_ch_2[0], tekubi[2])
            ren2_1 = sim_pearson(S_ch_2[1], tekubi[2])
            score2 = clf2.score(S_ch_2, tekubi[2])

            clf2.fit(S_ch_2, yubi[2])
            print(clf2.coef_)
            b12_0 = (clf2.coef_[0] * ((S0_2v / y_2v) ** 0.5)) ** 2
            b12_1 = (clf2.coef_[1] * ((S1_2v / y_2v) ** 0.5)) ** 2
            ren12 = sim_pearson(S_ch_2[0], yubi[2])
            ren12_1 = sim_pearson(S_ch_2[1], yubi[2])
            score12 = clf2.score(S_ch_2, yubi[2])
            if abs(ren2) > abs(ren12):
                s_t2 = S_ch_2[0]
                s_y2 = S_ch_2[1]
            else:
                s_t2 = S_ch_2[1]
                s_y2 = S_ch_2[0]

            ren_list2 = [ ren2, ren2_1, ren12, ren12_1 ]
            if ren_list2.index(max(ren_list2)) == 0 or ren_list2.index(max(ren_list2)) == 3:
                ren_list2 = [ ren_list2[ r ] for r in range(len(ren_list2)) if r == 0 or r == 3 ]
            else:
                ren_list2 = [ ren_list2[ r ] for r in range(len(ren_list2)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b2_0)
            print("標準化偏差回帰係数2")
            print(b2_1)
            print("S0の寄与率")
            print((b2_0 / (b2_0 + b2_1)) * 100)
            print("S1の寄与率")
            print((b2_1 / (b2_0 + b2_1)) * 100)
            print("相関係数")
            print(ren2)
            print(ren2_1)
            print("決定係数")
            print(score2)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b12_0)
            print("標準化偏差回帰係数2")
            print(b12_1)
            print("S0の寄与率")
            print((b12_0 / (b12_0 + b12_1)) * 100)
            print("S1の寄与率")
            print((b12_1 / (b12_0 + b12_1)) * 100)
            print("相関係数")
            print(ren12)
            print(ren12_1)
            print("決定係数")
            print(score12)
            print()

            print("////////////////////////////////////////////////////////")
            print("ch4")
            print()
            clf3.fit(S_ch_3, tekubi[3])
            print(clf3.coef_)
            b3_0 = (clf3.coef_[0] * ((S0_3v / t_3v) ** 0.5)) ** 2
            b3_1 = (clf3.coef_[1] * ((S1_3v / t_3v) ** 0.5)) ** 2
            ren3 = sim_pearson(S_ch_3[0], tekubi[3])
            ren3_1 = sim_pearson(S_ch_3[1], tekubi[3])
            score3 = clf3.score(S_ch_3, tekubi[3])
            clf3.fit(S_ch_3, yubi[3])
            print(clf3.coef_)
            b13_0 = (clf3.coef_[0] * ((S0_3v / y_3v) ** 0.5)) ** 2
            b13_1 = (clf3.coef_[1] * ((S1_3v / y_3v) ** 0.5)) ** 2
            ren13 = sim_pearson(S_ch_3[0], yubi[3])
            ren13_1 = sim_pearson(S_ch_3[1], yubi[3])
            score13 = clf3.score(S_ch_3, yubi[3])
            if abs(ren3) > abs(ren13):
                s_t3 = S_ch_3[0]
                s_y3 = S_ch_3[1]
            else:
                s_t3 = S_ch_3[1]
                s_y3 = S_ch_3[0]

            ren_list3 = [ ren3, ren3_1, ren13, ren13_1 ]
            if ren_list3.index(max(ren_list3)) == 0 or ren_list3.index(max(ren_list3)) == 3:
                ren_list3 = [ ren_list3[ r ] for r in range(len(ren_list3)) if r == 0 or r == 3 ]
            else:
                ren_list3 = [ ren_list3[ r ] for r in range(len(ren_list3)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b3_0)
            print("標準化偏差回帰係数2")
            print(b3_1)
            print("S0の寄与率")
            print((b3_0 / (b3_0 + b3_1)) * 100)
            print("S1の寄与率")
            print((b3_1 / (b3_0 + b3_1)) * 100)
            print("相関係数")
            print(ren3)
            print(ren3_1)
            print("決定係数")
            print(score3)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b13_0)
            print("標準化偏差回帰係数2")
            print(b13_1)
            print("S0の寄与率")
            print((b13_0 / (b13_0 + b13_1)) * 100)
            print("S1の寄与率")
            print((b13_1 / (b13_0 + b13_1)) * 100)
            print("相関係数")
            print(ren13)
            print(ren13_1)
            print("決定係数")
            print(score13)
            print()

            print("////////////////////////////////////////////////////////")
            print("ch5")
            print()
            clf4.fit(S_ch_4, tekubi[4])
            print(clf4.coef_)
            b4_0 = (clf4.coef_[0] * ((S0_4v / t_4v) ** 0.5)) ** 2
            b4_1 = (clf4.coef_[1] * ((S1_4v / t_4v) ** 0.5)) ** 2
            ren4 = sim_pearson(S_ch_4[0], tekubi[4])
            ren4_1 = sim_pearson(S_ch_4[1], tekubi[4])
            score4 = clf4.score(S_ch_4, tekubi[4])
            clf4.fit(S_ch_4, yubi[4])
            print(clf4.coef_)
            b14_0 = (clf4.coef_[0] * ((S0_4v / y_4v) ** 0.5)) ** 2
            b14_1 = (clf4.coef_[1] * ((S1_4v / y_4v) ** 0.5)) ** 2
            ren14 = sim_pearson(S_ch_4[0], yubi[4])
            ren14_1 = sim_pearson(S_ch_4[1], yubi[4])
            score14 = clf4.score(S_ch_4, yubi[4])
            if abs(ren4) > abs(ren14):
                s_t4 = S_ch_4[0]
                s_y4 = S_ch_4[1]
            else:
                s_t4 = S_ch_4[1]
                s_y4 = S_ch_4[0]

            ren_list4 = [ ren4, ren4_1, ren14, ren14_1 ]
            if ren_list4.index(max(ren_list4)) == 0 or ren_list4.index(max(ren_list4)) == 3:
                ren_list4 = [ ren_list4[ r ] for r in range(len(ren_list4)) if r == 0 or r == 3 ]
            else:
                ren_list4 = [ ren_list4[ r ] for r in range(len(ren_list4)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b4_0)
            print("標準化偏差回帰係数2")
            print(b4_1)
            print("S0の寄与率")
            print((b4_0 / (b4_0 + b4_1)) * 100)
            print("S1の寄与率")
            print((b4_1 / (b4_0 + b4_1)) * 100)
            print("相関係数")
            print(ren4)
            print(ren4_1)
            print("決定係数")
            print(score4)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b14_0)
            print("標準化偏差回帰係数2")
            print(b14_1)
            print("S0の寄与率")
            print((b14_0 / (b14_0 + b14_1)) * 100)
            print("S1の寄与率")
            print((b14_1 / (b14_0 + b14_1)) * 100)
            print("相関係数")
            print(ren14)
            print(ren14_1)
            print("決定係数")
            print(score14)
            print()
            print("////////////////////////////////////////////////////////")
            print("ch6")
            print()
            clf5.fit(S_ch_5, tekubi[5])
            print(clf5.coef_)
            b5_0 = (clf5.coef_[0] * ((S0_5v / t_5v) ** 0.5)) ** 2
            b5_1 = (clf5.coef_[1] * ((S1_5v / t_5v) ** 0.5)) ** 2
            ren5 = sim_pearson(S_ch_5[0], tekubi[5])
            ren5_1 = sim_pearson(S_ch_5[1], tekubi[5])
            score5 = clf5.score(S_ch_5, tekubi[5])
            clf5.fit(S_ch_5, yubi[5])
            print(clf5.coef_)
            b15_0 = (clf5.coef_[0] * ((S0_5v / y_5v) ** 0.5)) ** 2
            b15_1 = (clf5.coef_[1] * ((S1_5v / y_5v) ** 0.5)) ** 2
            ren15 = sim_pearson(S_ch_5[0], yubi[5])
            ren15_1 = sim_pearson(S_ch_5[1], yubi[5])
            score15 = clf5.score(S_ch_5, yubi[5])
            if abs(ren5) > abs(ren15):
                s_t5 = S_ch_5[0]
                s_y5 = S_ch_5[1]
            else:
                s_t5 = S_ch_5[1]
                s_y5 = S_ch_5[0]

            ren_list5 = [ ren5, ren5_1, ren15, ren15_1 ]
            if ren_list5.index(max(ren_list5)) == 0 or ren_list5.index(max(ren_list5)) == 3:
                ren_list5 = [ ren_list5[ r ] for r in range(len(ren_list5)) if r == 0 or r == 3 ]
            else:
                ren_list5 = [ ren_list5[ r ] for r in range(len(ren_list5)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b5_0)
            print("標準化偏差回帰係数2")
            print(b5_1)
            print("S0の寄与率")
            print((b5_0 / (b5_0 + b5_1)) * 100)
            print("S1の寄与率")
            print((b5_1 / (b5_0 + b5_1)) * 100)
            print("相関係数")
            print(ren5)
            print(ren5_1)
            print("決定係数")
            print(score5)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b15_0)
            print("標準化偏差回帰係数2")
            print(b15_1)
            print("S0の寄与率")
            print((b15_0 / (b15_0 + b15_1)) * 100)
            print("S1の寄与率")
            print((b15_1 / (b15_0 + b15_1)) * 100)
            print("相関係数")
            print(ren15)
            print(ren15_1)
            print("決定係数")
            print(score15)
            print()
            print("////////////////////////////////////////////////////////")
            print("ch7")
            print()
            clf6.fit(S_ch_6, tekubi[6])
            print(clf6.coef_)
            b6_0 = (clf6.coef_[0] * ((S0_6v / t_6v) ** 0.5)) ** 2
            b6_1 = (clf6.coef_[1] * ((S1_6v / t_6v) ** 0.5)) ** 2
            ren6 = sim_pearson(S_ch_6[0], tekubi[6])
            ren6_1 = sim_pearson(S_ch_6[1], tekubi[6])
            score6 = clf6.score(S_ch_6, tekubi[6])
            clf6.fit(S_ch_6, yubi[ 6 ])
            print(clf6.coef_)
            b16_0 = (clf6.coef_[ 0 ] * ((S0_6v / y_6v) ** 0.5)) ** 2
            b16_1 = (clf6.coef_[ 1 ] * ((S1_6v / y_6v) ** 0.5)) ** 2
            ren16 = sim_pearson(S_ch_6[ 0 ], yubi[ 6 ])
            ren16_1 = sim_pearson(S_ch_6[ 1 ], yubi[ 6 ])
            score16 = clf6.score(S_ch_6, yubi[ 6 ])
            if abs(ren6) > abs(ren16):
                s_t6 = S_ch_6[0]
                s_y6 = S_ch_6[1]
            else:
                s_t6 = S_ch_6[1]
                s_y6 = S_ch_6[0]

            ren_list6 = [ ren6, ren6_1, ren16, ren16_1 ]
            if ren_list6.index(max(ren_list6)) == 0 or ren_list6.index(max(ren_list6)) == 3:
                ren_list6 = [ ren_list6[ r ] for r in range(len(ren_list6)) if r == 0 or r == 3 ]
            else:
                ren_list6 = [ ren_list6[ r ] for r in range(len(ren_list6)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b6_0)
            print("標準化偏差回帰係数2")
            print(b6_1)
            print("S0の寄与率")
            print((b6_0 / (b6_0 + b6_1)) * 100)
            print("S1の寄与率")
            print((b6_1 / (b6_0 + b6_1)) * 100)
            print("相関係数")
            print(ren6)
            print(ren6_1)
            print("決定係数")
            print(score6)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b16_0)
            print("標準化偏差回帰係数2")
            print(b16_1)
            print("S0の寄与率")
            print((b16_0 / (b16_0 + b16_1)) * 100)
            print("S1の寄与率")
            print((b16_1 / (b16_0 + b16_1)) * 100)
            print("相関係数")
            print(ren16)
            print(ren16_1)
            print("決定係数")
            print(score16)
            print()
            print("////////////////////////////////////////////////////////")
            print("ch8")
            print()
            clf7.fit(S_ch_7, tekubi[7])
            print(clf7.coef_)
            b7_0 = (clf7.coef_[0] * ((S0_7v / t_7v) ** 0.5)) ** 2
            b7_1 = (clf7.coef_[1] * ((S1_7v / t_7v) ** 0.5)) ** 2
            ren7 = sim_pearson(S_ch_7[0], tekubi[7])
            ren7_1 = sim_pearson(S_ch_7[1], tekubi[7])
            score7 = clf7.score(S_ch_7, tekubi[7])
            clf7.fit(S_ch_7, yubi[7])
            print(clf7.coef_)
            b17_0 = (clf7.coef_[0] * ((S0_7v / y_7v) ** 0.5)) ** 2
            b17_1 = (clf7.coef_[1] * ((S1_7v / y_7v) ** 0.5)) ** 2
            ren17 = sim_pearson(S_ch_7[0], yubi[7])
            ren17_1 = sim_pearson(S_ch_7[1], yubi[7])
            score17 = clf7.score(S_ch_7, yubi[7])
            if abs(ren7) > abs(ren17):
                s_t7 = S_ch_7[0]
                s_y7 = S_ch_7[1]
            else:
                s_t7 = S_ch_7[1]
                s_y7 = S_ch_7[0]

            ren_list7 = [ ren7, ren7_1, ren17, ren17_1 ]
            if ren_list7.index(max(ren_list7)) == 0 or ren_list7.index(max(ren_list7)) == 3:
                ren_list7 = [ ren_list7[ r ] for r in range(len(ren_list7)) if r == 0 or r == 3 ]
            else:
                ren_list7 = [ ren_list7[ r ] for r in range(len(ren_list7)) if r == 1 or r == 2 ]

            print()
            print("//////// 手首 ////////")
            print("標準化偏差回帰係数1")
            print(b7_0)
            print("標準化偏差回帰係数2")
            print(b7_1)
            print("S0の寄与率")
            print((b7_0 / (b7_0 + b7_1)) * 100)
            print("S1の寄与率")
            print((b7_1 / (b7_0 + b7_1)) * 100)
            print("相関係数")
            print(ren7)
            print(ren7_1)
            print("決定係数")
            print(score7)
            print()
            print("//////// 指 ////////")
            print("標準化偏差回帰係数1")
            print(b17_0)
            print("標準化偏差回帰係数2")
            print(b17_1)
            print("S0の寄与率")
            print((b17_0 / (b17_0 + b17_1)) * 100)
            print("S1の寄与率")
            print((b17_1 / (b17_0 + b17_1)) * 100)
            print("相関係数")
            print(ren17)
            print(ren17_1)
            print("決定係数")
            print(score17)
            print()
            print("////////////////////////////////////////////////////////")
            print()
            nf = open(path.png_ica_new + "/" + i + "/" + d + "/" + s + "/" + "fastICA" + "/" + "結果_fastICA.CSV", 'w', encoding="utf_8_sig")
            dataWriter = csv.writer(nf)
            dataWriter.writerow(["ch1"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b0_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b0_1])
            dataWriter.writerow(["", "S0の寄与率", (b0_0 / (b0_0 + b0_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b0_1 / (b0_0 + b0_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren0])
            dataWriter.writerow(["", "相関係数2", ren0_1])
            dataWriter.writerow(["", "決定係数", score0])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b10_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b10_1])
            dataWriter.writerow(["", "S0の寄与率", (b10_0 / (b10_0 + b10_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b10_1 / (b10_0 + b10_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren10])
            dataWriter.writerow(["", "相関係数2", ren10_1])
            dataWriter.writerow(["", "決定係数", score10])
            dataWriter.writerow([])
            dataWriter.writerow(["ch2"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b1_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b1_1])
            dataWriter.writerow(["", "S0の寄与率", (b1_0 / (b1_0 + b1_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b1_1 / (b1_0 + b1_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren1])
            dataWriter.writerow(["", "相関係数2", ren1_1])
            dataWriter.writerow(["", "決定係数", score1])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b11_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b11_1])
            dataWriter.writerow(["", "S0の寄与率", (b11_0 / (b11_0 + b11_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b11_1 / (b11_0 + b11_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren11])
            dataWriter.writerow(["", "相関係数2", ren11_1])
            dataWriter.writerow(["", "決定係数", score11])
            dataWriter.writerow([])
            dataWriter.writerow(["ch3"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b2_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b2_1])
            dataWriter.writerow(["", "S0の寄与率", (b2_0 / (b2_0 + b2_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b2_1 / (b2_0 + b2_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren2])
            dataWriter.writerow(["", "相関係数2", ren2_1])
            dataWriter.writerow(["", "決定係数", score2])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b12_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b12_1])
            dataWriter.writerow(["", "S0の寄与率", (b12_0 / (b12_0 + b12_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b12_1 / (b12_0 + b12_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren12])
            dataWriter.writerow(["", "相関係数2", ren12_1])
            dataWriter.writerow(["", "決定係数", score12])
            dataWriter.writerow([])
            dataWriter.writerow(["ch4"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b3_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b3_1])
            dataWriter.writerow(["", "S0の寄与率", (b3_0 / (b3_0 + b3_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b3_1 / (b3_0 + b3_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren3])
            dataWriter.writerow(["", "相関係数2", ren3_1])
            dataWriter.writerow(["", "決定係数", score3])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b13_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b13_1])
            dataWriter.writerow(["", "S0の寄与率", (b13_0 / (b13_0 + b13_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b13_1 / (b13_0 + b13_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren13])
            dataWriter.writerow(["", "相関係数2", ren13_1])
            dataWriter.writerow(["", "決定係数", score13])
            dataWriter.writerow([])
            dataWriter.writerow(["ch5"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b4_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b4_1])
            dataWriter.writerow(["", "S0の寄与率", (b4_0 / (b4_0 + b4_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b4_1 / (b4_0 + b4_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren4])
            dataWriter.writerow(["", "相関係数2", ren4_1])
            dataWriter.writerow(["", "決定係数", score4])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b14_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b14_1])
            dataWriter.writerow(["", "S0の寄与率", (b14_0 / (b14_0 + b14_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b14_1 / (b14_0 + b14_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren14])
            dataWriter.writerow(["", "相関係数2", ren14_1])
            dataWriter.writerow(["", "決定係数", score14])
            dataWriter.writerow([])
            dataWriter.writerow(["ch6"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b5_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b5_1])
            dataWriter.writerow(["", "S0の寄与率", (b5_0 / (b5_0 + b5_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b5_1 / (b5_0 + b5_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren5])
            dataWriter.writerow(["", "相関係数2", ren5_1])
            dataWriter.writerow(["", "決定係数", score5])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b15_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b15_1])
            dataWriter.writerow(["", "S0の寄与率", (b15_0 / (b15_0 + b15_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b15_1 / (b15_0 + b15_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren15])
            dataWriter.writerow(["", "相関係数2", ren15_1])
            dataWriter.writerow(["", "決定係数", score15])
            dataWriter.writerow([])
            dataWriter.writerow(["ch7"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b6_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b6_1])
            dataWriter.writerow(["", "S0の寄与率", (b6_0 / (b6_0 + b6_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b6_1 / (b6_0 + b6_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren6])
            dataWriter.writerow(["", "相関係数2", ren6_1])
            dataWriter.writerow(["", "決定係数", score6])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b16_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b16_1])
            dataWriter.writerow(["", "S0の寄与率", (b16_0 / (b16_0 + b16_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b16_1 / (b16_0 + b16_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren16])
            dataWriter.writerow(["", "相関係数2", ren16_1])
            dataWriter.writerow(["", "決定係数", score16])
            dataWriter.writerow([])
            dataWriter.writerow(["ch8"])
            dataWriter.writerow(["手首", "標準化偏差回帰係数1", b7_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b7_1])
            dataWriter.writerow(["", "S0の寄与率", (b7_0 / (b7_0 + b7_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b7_1 / (b7_0 + b7_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren7])
            dataWriter.writerow(["", "相関係数2", ren7_1])
            dataWriter.writerow(["", "決定係数", score7])
            dataWriter.writerow(["指", "標準化偏差回帰係数1", b17_0])
            dataWriter.writerow(["", "標準化偏差回帰係数2", b17_1])
            dataWriter.writerow(["", "S0の寄与率", (b17_0 / (b17_0 + b17_1)) * 100])
            dataWriter.writerow(["", "S1の寄与率", (b17_1 / (b17_0 + b17_1)) * 100])
            dataWriter.writerow(["", "相関係数1", ren17])
            dataWriter.writerow(["", "相関係数2", ren17_1])
            dataWriter.writerow(["", "決定係数", score17])
            nf.close()
            ren_list_avarage_0 = (ren_list0[0] + ren_list1[0] + ren_list2[0] + ren_list3[0] + ren_list4[0] +
                                  ren_list5[0] + ren_list6[0] + ren_list7[0]) / 8
            ren_list_avarage_1 = (ren_list0[1] + ren_list1[1] + ren_list2[1] + ren_list3[1] + ren_list4[1] +
                                  ren_list5[1] + ren_list6[1] + ren_list7[1]) / 8
            ren_list_avarage_all = (ren_list_avarage_0 + ren_list_avarage_1) / 2

            dataWriter_all.writerow(["手首", ren_list0[0], ren_list1[0], ren_list2[0], ren_list3[0], ren_list4[0],
                                     ren_list5[0], ren_list6[0], ren_list7[0], ren_list_avarage_0, ren_list_avarage_all])
            dataWriter_all.writerow(["指", ren_list0[1], ren_list1[1], ren_list2[1], ren_list3[1], ren_list4[1],
                                     ren_list5[1], ren_list6[1], ren_list7[1], ren_list_avarage_1])

            # 相関係数
            # if fft_flg == 1:
            #     tekubi_ch_list_p = pd.DataFrame(tekubi_ch_list_abs)
            #     yubi_ch_list_p = pd.DataFrame(yubi_ch_list_abs)
            #     S_0_ch_list_p = pd.DataFrame(S_0_ch_list_abs)
            #     S_1_ch_list_p = pd.DataFrame(S_1_ch_list_abs)
            #     mix_ch_list_p = pd.DataFrame(mix_ch_list)
            #     mix_2_ch_list_p = pd.DataFrame(mix_2_ch_list)
            #     X_0_ch_list_p = pd.DataFrame(X_0_ch_list_abs)
            #     X_1_ch_list_p = pd.DataFrame(X_1_ch_list_abs)
            #
            # else:
            tekubi_ch_list_p = pd.DataFrame(tekubi_ch_list_m)
            yubi_ch_list_p = pd.DataFrame(yubi_ch_list_m)
            #     S_0_ch_list_p = pd.DataFrame(S_0_ch_list_m)
            #     S_1_ch_list_p = pd.DataFrame(S_1_ch_list_m)
            #     mix_ch_list_p = pd.DataFrame(mix_ch_list)
            #     mix_2_ch_list_p = pd.DataFrame(mix_2_ch_list)
            #     X_0_ch_list_p = pd.DataFrame(X_0_ch_list_m)
            #     X_1_ch_list_p = pd.DataFrame(X_1_ch_list_m)
            #
            res_ = tekubi_ch_list_p.T.corrwith(yubi_ch_list_p.T)
            print(res_)
            # print(sum(res)/7)

            s_tekubi_list = [s_t0, s_t1, s_t2, s_t3, s_t4, s_t5, s_t6, s_t7]
            s_yubi_list = [s_y0, s_y1, s_y2, s_y3, s_y4, s_y5, s_y6, s_y7]
            # s_tekubi_list = [s_t0, s_t1, s_t2, s_t3]
            # s_yubi_list = [s_y0, s_y1, s_y2, s_y3]


            if debug == 1 or debug == 3:
                for m in range(0, 8):

                        # plot_mixture_sources_predictions(X_ch_list_abs[i], [yubi_ch_list_abs[i], tekubi_ch_list_abs[i]],
                        #                                  S_ch_list_abs[i], i + 1)
                        plot_mixture_sources_predictions([ mix_ch_list_m[ m ], mix_2_ch_list_m[ m ] ],
                                                         [ tekubi_ch_list_m[ m ], yubi_ch_list_m[ m ] ],
                                                         [ s_tekubi_list[ m ], s_yubi_list[ m ] ], m + 1)

                        # plot_mixture_sources_predictions(X_ch_list[i], [yubi_ch_list[i], tekubi_ch_list[i]], S_ch_list[i], i + 1)
                        #plot_mixture_sources_predictions([mix_ch_list_m[i], mix_2_ch_list_m[i]], [tekubi_ch_list_m[i], yubi_ch_list_m[i]], [S_0_ch_list_m[i], S_1_ch_list_m[i]], i + 1)

            if debug == 2 or debug == 3:
                write_plot(tekubi_ch_list_m, str("tekubi"))
                write_plot(mix_2_ch_list_m, str("mix1"))
                write_plot(mix_ch_list_m, str("mix2"))
                write_plot(yubi_ch_list_m, str("yubi"))
                write_plot(s_tekubi_list, str("s_tekubi"))
                write_plot(s_yubi_list, str("s_yubi"))

            sub_average += ren_list_avarage_all
    dataWriter_all.writerow(["", "", "", "", "", "", "", "", "", "", sub_average/12])
    all_average += sub_average/12
dataWriter_all.writerow([])
dataWriter_all.writerow(["", "", "", "", "", "", "", "", "", "", all_average/3])
nf_all.close()