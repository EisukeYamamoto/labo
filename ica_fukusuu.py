import matplotlib.pyplot as plt
from app import ica
import pandas as pd
import numpy as np
import math
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from app import path
from sklearn import linear_model
from sklearn import preprocessing

clf0 = linear_model.LinearRegression()

sscaler = preprocessing.StandardScaler()


fft_flg = 2
plot_flg = 0
clf_flg = 1


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


def center(x):
    x = np.array(x)

    mean = x.mean(axis=0, keepdims=True)

    return x - mean


def whitening(x):
    cov = np.cov(x)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.pinv(D))
    x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x)))
    return x_whiten


def calculate_new_w(w, x):
    w_new = (x * g(np.dot(w.T, x))).mean(axis=1) - g_der(np.dot(w.T, x)).mean() * w
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
            # print(distance)

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

def plot_mixture_sources_predictions(x, original_source, s):
    fig = plt.figure()

    plt.subplot(3, 1, 1)
    for x_val in x:
        plt.plot(x_val)

    plt.title("mixtures")

    plt.subplot(3, 1, 2)
    for ori_val in original_source:
        plt.plot(ori_val)
    plt.title("real sources")

    plt.subplot(3, 1, 3)
    for s_val in s:
        plt.plot(s_val)
    plt.title("predicted sources")

    fig.tight_layout()

    plt.show()


def write_plot(ch_list, name):
    # print("test/" + d + "/" + i + "/" + j + "/" + l + ".CSV/")
    len_num = len(ch_list[0])
    x = np.linspace(0, len_num, len_num)
    y0 = ch_list[0]
    y1 = ch_list[1]
    y2 = ch_list[2]
    y3 = ch_list[3]
    # y4 = ch_list[4] #
    # y5 = ch_list[5] #
    # y6 = ch_list[6] #

    labels0 = [-0.5, 0.5]
    labels1 = [-0.5, 0.5]
    labels2 = [-0.5, 0.5]
    labels3 = [-0.5, 0.5]
    # labels4 = [-0.5, 0.5] #
    # labels5 = [-0.5, 0.5] #
    # labels6 = [-0.5, 0.5] #

    # l0, l1, l2, l3, l4, l5, l6 = "ch1", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8" #
    # o0, o1, o2, o3, o4, o5, o6 = 1, 3, 5, 7, 9, 11, 13 #
    l0, l1, l2, l3 = "ch1", "ch2", "ch3", "ch4"
    o0, o1, o2, o3 = 1, 3, 5, 7
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
    # yticks4 = [la + o4 for la in labels4] #
    # yticks5 = [la + o5 for la in labels5] #
    # yticks6 = [la + o6 for la in labels6] #

    # ytls = labels0 + labels1 + labels2 + labels3 + labels4 + labels5 + labels6 #
    # ytks = yticks0 + yticks1 + yticks2 + yticks3 + yticks4 + yticks5 + yticks6 #
    # plt.figure(figsize=(6, 5), facecolor="w") #
    # yo0 = list(map(lambda x: x + o6, y0)) #
    # yo1 = list(map(lambda x: x + o5, y1)) #
    # yo2 = list(map(lambda x: x + o4, y2)) #
    # yo3 = list(map(lambda x: x + o3, y3)) #
    # yo4 = list(map(lambda x: x + o2, y4)) #
    # yo5 = list(map(lambda x: x + o1, y5)) #
    # yo6 = list(map(lambda x: x + o0, y6)) #
    #
    # plt.plot(x, yo0, color="b", label=l0) #
    # plt.plot(x, yo1, color="r", label=l1) #
    # plt.plot(x, yo2, color="g", label=l2) #
    # plt.plot(x, yo3, color="c", label=l3) #
    # plt.plot(x, yo4, color="m", label=l4) #
    # plt.plot(x, yo5, color="y", label=l5) #
    # plt.plot(x, yo6, color="k", label=l6) #

    ytls = labels0 + labels1 + labels2 + labels3
    ytks = yticks0 + yticks1 + yticks2 + yticks3
    plt.figure(figsize=(6, 5), facecolor="w")
    yo0 = list(map(lambda x: x + o3, y0))
    yo1 = list(map(lambda x: x + o2, y1))
    yo2 = list(map(lambda x: x + o1, y2))
    yo3 = list(map(lambda x: x + o0, y3))

    plt.plot(x, yo0, color="b", label=l0)
    plt.plot(x, yo1, color="r", label=l1)
    plt.plot(x, yo2, color="g", label=l2)
    plt.plot(x, yo3, color="c", label=l3)

    plt.ylim(o0 - 2, o3 + 2)
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
        # plotpath2 = path.png_ica + "/" + d + "/" + i + "/複合+複合2/" + name
        plotpath2 = path.png_ica + "/" + d + "/" + i + "/" + name
        plt.savefig(plotpath2)
        plt.show()
    else: plt.show()
    plt.close('all')


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):
        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    x = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        x += 0.002 * np.random.normal(size=x.shape)

    return x


def max_ch(l):
    l_min = min(l)
    l_max = max(l)
    return [l_max - l_min, l_min]


def min_max(l, r, min_):
    print(min_)
    return [(i - min_) / r for i in l]


def mapping(y1, y2):
    # print(y1.shape)
    r1 = max_ch(y1)
    r2 = max_ch(y2)
    max_list = [r1[0], r2[0]]
    range_ = max(max_list)
    y1 = center(min_max(y1, range_, min(y1)))
    y2 = center(min_max(y2, range_, min(y2)))

    y1_max = max_ch(y1)
    y2_max = max_ch(y2)
    max_list2 = [y1_max[0], y2_max[0]]
    range_2 = max(max_list2)
    y1 = min_max(y1, range_2, min(y1))
    y2 = min_max(y2, range_2, min(y2))

    new_ch_list = [y1, y2]

    return new_ch_list


def mapping_ssc(y):
    len_num = len(y)
    x = np.linspace(0, len_num, len_num)

    sscaler.fit(y)
    new_ch_list = sscaler.transform(y)

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

    if variance1 * variance2 == 0:
        return 0

    return covariance / (variance1 * variance2)


def pca_change(value):
    pca = PCA(n_components=1)
    new_value = np.array(pca.fit_transform(value.T))
    print('各主成分の寄与率:', pca.explained_variance_ratio_)
    print('寄与率の累積:', sum(pca.explained_variance_ratio_))
    # new_value = new_value.T
    # print(new_value.shape)

    return new_value


# 複合動作
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.mix_only:
            # for o in path.cut8_times:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
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
                    # mix_ch4 = mix1.iloc[:, 4].values #
                    # mix_ch5 = mix1.iloc[:, 5].values #
                    # mix_ch6 = mix1.iloc[:, 6].values #

                    print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")

                    feature_mix1_value = []
                    feature_mix1_value.append(mix_ch0)
                    feature_mix1_value.append(mix_ch1)
                    feature_mix1_value.append(mix_ch2)
                    feature_mix1_value.append(mix_ch3)
                    # feature_mix1_value.extend(mix_ch4)
                    # feature_mix1_value.extend(mix_ch5)
                    # feature_mix1_value.extend(mix_ch6)
                    # print(len(feature_mix1_value))
                    mix1_value = pd.DataFrame(feature_mix1_value)
                    # print(mix1_value.shape)

# # 複合動作2
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.mix_only2:
            # for o in path.cut8_times:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
                    elif fft_flg == 2:
                        file_path = path.plus_avarage_fft_long + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
                    else:
                        file_path = path.plus_avarage_ifft + "/" + d + "/" + i + "/" + j + "/" + l + "ifft.CSV"
                    print(file_path)
                    mix2 = pd.read_csv(file_path)
                    # mix2 = outlier_iqr(mix2)

                    mix_2_ch0 = mix2.iloc[:, 0].values
                    mix_2_ch1 = mix2.iloc[:, 1].values
                    mix_2_ch2 = mix2.iloc[:, 2].values
                    mix_2_ch3 = mix2.iloc[:, 3].values
                    # mix_2_ch4 = mix2.iloc[:, 4].values #
                    # mix_2_ch5 = mix2.iloc[:, 5].values #
                    # mix_2_ch6 = mix2.iloc[:, 6].values #

                    print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")

                    feature_mix2_value = [ ]
                    feature_mix2_value.append(mix_2_ch0)
                    feature_mix2_value.append(mix_2_ch1)
                    feature_mix2_value.append(mix_2_ch2)
                    feature_mix2_value.append(mix_2_ch3)
                    # feature_mix2_value.extend(mix_2_ch4)
                    # feature_mix2_value.extend(mix_2_ch5)
                    # feature_mix2_value.extend(mix_2_ch6)
                    # print(len(feature_mix2_value))
                    mix2_value = pd.DataFrame(feature_mix2_value)
                    # print(mix2_value.shape)

# 手首動作
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.tekubi_only:
            # for o in path.cut8_times:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
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
                    # tekubi_ch4 = tekubi.iloc[:, 4].values #
                    # tekubi_ch5 = tekubi.iloc[:, 5].values #
                    # tekubi_ch6 = tekubi.iloc[:, 6].values #

                    print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")

                    feature_tekubi_value = [ ]
                    feature_tekubi_value.append(tekubi_ch0)
                    feature_tekubi_value.append(tekubi_ch1)
                    feature_tekubi_value.append(tekubi_ch2)
                    feature_tekubi_value.append(tekubi_ch3)
                    # feature_tekubi_value.extend(tekubi_ch4)
                    # feature_tekubi_value.extend(tekibi_ch5)
                    # feature_tekubi_value.extend(tekubi_ch6)
                    # print(len(feature_tekubi_value))
                    tekubi_value = pd.DataFrame(feature_tekubi_value)
                    # print(tekubi_value.shape)

# 指動作
for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.yubi_only:
            # for o in path.cut8_times:
                for l in path.times:
                    if fft_flg == 1:
                        file_path = path.plus_avarage_fft + "/" + d + "/" + i + "/" + j + "/" + l + "fft.CSV"
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
                    # yubi_ch4 = yubi.iloc[:, 4].values #
                    # yubi_ch5 = yubi.iloc[:, 5].values #
                    # yubi_ch6 = yubi.iloc[:, 6].values #

                    print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")

                    feature_yubi_value = [ ]
                    feature_yubi_value.append(yubi_ch0)
                    feature_yubi_value.append(yubi_ch1)
                    feature_yubi_value.append(yubi_ch2)
                    feature_yubi_value.append(yubi_ch3)
                    # feature_yubi_value.extend(yubi_ch4)
                    # feature_yubi_value.extend(yubi_ch5)
                    # feature_yubi_value.extend(yubi_ch6)
                    # print(len(feature_yubi_value))
                    yubi_value = pd.DataFrame(feature_yubi_value)
                    # print(yubi_value.shape)


mix1_p = np.ravel(pca_change(mix1_value))
mix2_p = np.ravel(pca_change(mix2_value))
tekubi_p = np.ravel(pca_change(tekubi_value))
yubi_p = np.ravel(pca_change(yubi_value))

X = mix_sources([mix1_p, mix2_p])

S = ica(X, iterations=1000)

mix1_p_abs = abs(mix1_p)**2
mix2_p_abs = abs(mix2_p)**2
tekubi_p_abs = abs(tekubi_p)**2
yubi_p_abs = abs(yubi_p)**2
X_abs = abs(X)**2
S_abs = abs(S)**2

if fft_flg == 0:
    origin_list_m = mapping(yubi_p, tekubi_p)
    S_list = mapping(S[0], S[1])
    mix_list = mapping(X[0], X[1])
    # mix_list = mapping_1(mix_ch_list, mix_2_ch_list)

else:
    origin_list_m = mapping(yubi_p_abs, tekubi_p_abs)
    S_list = mapping(S_abs[0], S_abs[1])
    mix_list = mapping(X_abs[0], X_abs[1])
    # mix_list = mapping_1(mix_ch_list_abs, mix_2_ch_list_abs)
    # mix_list_abs = mapping_1(mix_ch_list, mix_2_ch_list)

yubi_ch_list_m = origin_list_m[0]
tekubi_ch_list_m = origin_list_m[1]
S_0_ch_list_m = S_list[0]
S_1_ch_list_m = S_list[1]
mix_ch_list_m = mix_list[0]
mix_2_ch_list_m = mix_list[1]

# print("S_list")
# print(S_list)
# print(S_0_ch_list_m.shape)

y_p = pd.DataFrame(yubi_ch_list_m)
y_v = y_p.var().values

t_p = pd.DataFrame(tekubi_ch_list_m)
t_v = t_p.var().values

S0_p = pd.DataFrame(S_0_ch_list_m)
S0_v = S0_p.var().values

S1_p = pd.DataFrame(S_1_ch_list_m)
S1_v = S1_p.var().values

Mix_p = pd.DataFrame(mix_ch_list_m)
Mix_v = Mix_p.var().values

S_0p = pd.DataFrame(S_0_ch_list_m)
S_0_new = S_0p

S_1p = pd.DataFrame(S_1_ch_list_m)
S_1_new = S_1p

S_ch = []
S_ch.append(S_0_ch_list_m)
S_ch.append(S_1_ch_list_m)
S_ch = pd.DataFrame(S_ch)
S_ch = S_ch.T

tekubi_p = pd.DataFrame(tekubi_ch_list_m)
tekubi = tekubi_p
print("tekubi")
print(tekubi)

yubi_p = pd.DataFrame(yubi_ch_list_m)
yubi = yubi_p

X_0_p = pd.DataFrame(mix_ch_list_m)
X_0 = X_0_p

X_1_p = pd.DataFrame(mix_2_ch_list_m)
X_1 = X_1_p

mix_p = pd.DataFrame(mix_ch_list_m)
mix_ = mix_p

# print(mix_.shape)



if clf_flg == 1:
    print()
    print("////////////////////////////////////////////////////////")
    print("ch1")
    print()
    clf0.fit(S_ch, tekubi[0])
    print(clf0.coef_)
    b0_0 = (clf0.coef_[0] * ((S0_v / t_v) ** 0.5)) ** 2
    b0_1 = (clf0.coef_[1] * ((S1_v / t_v) ** 0.5)) ** 2
    ren0 = sim_pearson(S_ch[0], tekubi[0])
    ren0_1 = sim_pearson(S_ch[1], tekubi[0])
    score0 = clf0.score(S_ch, tekubi[0])

    clf0.fit(S_ch, yubi[0])
    print(clf0.coef_)
    b10_0 = (clf0.coef_[ 0 ] * ((S0_v / y_v) ** 0.5)) ** 2
    b10_1 = (clf0.coef_[ 1 ] * ((S1_v / y_v) ** 0.5)) ** 2
    ren10 = sim_pearson(S_ch[0], yubi[0])
    ren10_1 = sim_pearson(S_ch[1], yubi[0])
    score10 = clf0.score(S_ch, yubi[0])
    if abs(ren0) > abs(ren10):
        s_t0 = S_ch[0]
        s_y0 = S_ch[1]
    else:
        s_t0 = S_ch[1]
        s_y0 = S_ch[0]

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

s_tekubi_list = [s_t0]
s_yubi_list = [s_y0]


plot_mixture_sources_predictions([mix_ch_list_m, mix_2_ch_list_m],
                                [tekubi_ch_list_m, yubi_ch_list_m],
                                [s_tekubi_list[0], s_yubi_list[0]])