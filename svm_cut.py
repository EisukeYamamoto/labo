# coding:utf-8
"""
svm
"""
from app import path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

N = 127
dt = 0.01
fc = 20


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

                        if (j == '指'):
                            test_y.append('指')
                        elif (j == '手首'):
                            test_y.append('手首')


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

                        if (j == '指'):
                            train_y.append('指')
                        elif (j == '手首'):
                            train_y.append('手首')


    # svmで識別
    # データの標準化
    sc = StandardScaler()
    x_std_train = sc.fit_transform(X=train_x)
    x_std_test = sc.transform(X=test_x)

    accuracy_list = []
    parameter_list = []

    for c in cost_value_list:
        for g in gamma_value_list:
            predict_list = []

            # 線形SVM
            svm = SVC(C=c, kernel="rbf", gamma=g)
            svm.fit(X=x_std_train, y=train_y)

            for i in range(0, len(x_std_test)):
                predict_list.append(svm.predict(X=[x_std_test[i]]))

            print()
            print()
            # print(x_std_test.shape)
            print(predict_list)
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


if __name__ == '__main__':
    svm()
