# coding:utf-8
"""
svm
"""
from app import path
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

N = 127    # データ数
dt = 0.01  # サンプリング周期
fc = 20    # カットオフ(周波数)
p = 80     # 主成分分析(次元数)

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
        a[a < outlier_min] = None
        a[a > outlier_max] = None

    return df

def svm():
    test_x = []
    test_y = []
    train_x = []
    train_y = []

    cost_value_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    gamma_value_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    # テストファイルの読み込み
    for d in path.test_day:
        for i in path.subject:
            for j in path.char:
                for o in path.cut_time:
                    for l in path.test_time:
                        # file_path = path.cut8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        # file_path = path.fft_8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "fft.CSV"
                        file_path = path.ifft_8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "ifft.CSV"
                        df = pd.read_csv(file_path)

                        print("test/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                        # print(df)

                        ch0 = list(df.iloc[:, 0].values)
                        ch1 = list(df.iloc[:, 1].values)
                        ch2 = list(df.iloc[:, 2].values)
                        ch3 = list(df.iloc[:, 3].values)
                        ch4 = list(df.iloc[:, 4].values)
                        ch5 = list(df.iloc[:, 5].values)
                        ch6 = list(df.iloc[:, 6].values)
                        # ch7 = list(df.iloc[:, 7].values)

                        feature_value = []
                        feature_value.extend(ch0)
                        feature_value.extend(ch1)
                        feature_value.extend(ch2)
                        feature_value.extend(ch3)
                        feature_value.extend(ch4)
                        feature_value.extend(ch5)
                        feature_value.extend(ch6)
                        # feature_value.extend(ch7)

                        test_x.append(feature_value)

                        if (i == 'sub_A'):
                            test_y.append('sub_A')
                        elif (i == 'sub_B'):
                            test_y.append('sub_B')
                        elif (i == 'sub_C'):
                            test_y.append('sub_C')


    # ファイルの読み込み
    for d in path.test_day:
        for i in path.subject:
            for j in path.char:
                for o in path.cut_time:
                    for l in path.train_time:
                        # file_path = path.cut8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        # file_path = path.fft_8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "fft.CSV"
                        file_path = path.ifft_8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "ifft.CSV"
                        df = pd.read_csv(file_path)
                        print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                        # print(df)

                        ch0 = list(df.iloc[:, 0].values)
                        ch1 = list(df.iloc[:, 1].values)
                        ch2 = list(df.iloc[:, 2].values)
                        ch3 = list(df.iloc[:, 3].values)
                        ch4 = list(df.iloc[:, 4].values)
                        ch5 = list(df.iloc[:, 5].values)
                        ch6 = list(df.iloc[:, 6].values)
                        # ch7 = list(df.iloc[:, 7].values)

                        feature_value = []
                        feature_value.extend(ch0)
                        feature_value.extend(ch1)
                        feature_value.extend(ch2)
                        feature_value.extend(ch3)
                        feature_value.extend(ch4)
                        feature_value.extend(ch5)
                        feature_value.extend(ch6)
                        # feature_value.extend(ch7)

                        train_x.append(feature_value)

                        if (i == 'sub_A'):
                            train_y.append('sub_A')
                        elif (i == 'sub_B'):
                            train_y.append('sub_B')
                        elif (i == 'sub_C'):
                            train_y.append('sub_C')


    # svmで識別
    # データの標準化
    sc = StandardScaler()
    x_std_train = sc.fit_transform(X=train_x)
    x_std_test = sc.transform(X=test_x)

    # 約8000次元から8次元に圧縮
    # n_componentsで次元変更
    pca = PCA(n_components=p)
    # 標準化したxデータをPCAで次元圧縮
    X_std_train_pca = pd.DataFrame(pca.fit_transform(x_std_train))
    X_std_test_pca = pd.DataFrame(pca.fit_transform(x_std_test))

    # 外れ値
    z_train = outlier_iqr(X_std_train_pca)
    z_test = outlier_iqr(X_std_test_pca)

    z_train_p = pd.DataFrame(z_train)
    train_i = np.array(z_train_p.index[z_train_p.isnull().any(axis=1) == True])
    print(train_i)
    print(train_y)
    i_2 = 0
    for i in train_i:
        # print(i)
        # print(train_y[i - i_2])
        del train_y[i - i_2]
        i_2 += 1

    print(train_y)
    z_train_n = np.array(z_train_p.dropna(how='any'))

    z_test_p = pd.DataFrame(z_test)
    test_i = np.array(z_test_p.index[z_test_p.isnull().any(axis=1) == True])
    print(test_i)
    print(test_y)

    i_2 = 0
    for i in test_i:
        # print(i)
        # print(test_y[i - i_2])
        del test_y[i - i_2]
        i_2 += 1
    print(test_y)
    z_test_n = np.array(z_test_p.dropna(how='any'))
    # print(z_train_n)

    accuracy_list = []
    parameter_list = []

    for c in cost_value_list:
        for g in gamma_value_list:
            predict_list = []

            # 線形SVM
            svm = SVC(C=c, kernel="rbf", gamma=g)
            svm.fit(X=z_train_n, y=train_y)

            for i in range(0, len(z_test_n)):
                predict_list.append(svm.predict(X=[z_test_n[i]]))

            predict_list_n = np.array(predict_list)

            print()
            print()
            # print(predict_list)
            print("パラメタ")
            print("c=" + str(c) + ", gamma=" + str(g))
            parameter_list.append("c=" + str(c) + ", gamma=" + str(g))
            print("算出頻度")
            print(accuracy_score(y_pred=predict_list, y_true=test_y))
            accuracy_list.append(accuracy_score(y_pred=predict_list, y_true=test_y))
            print("混同行列は")
            print(confusion_matrix(y_pred=predict_list, y_true=test_y))

    print("最高精度" + str(max(accuracy_list)))
    index = accuracy_list.index(max(accuracy_list))
    print("設定は")
    print(parameter_list[index])
    print('各主成分の寄与率:', pca.explained_variance_ratio_)
    print('寄与率の累積:', sum(pca.explained_variance_ratio_))


if __name__ == '__main__':
    svm()