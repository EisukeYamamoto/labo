# coding:utf-8

from matplotlib import pyplot as plt
from app import path
import pandas as pd
import numpy as np


def write_plot(ch_list, d, i, j, o, l, df):
    for n in range(0, 7):
        print("test/" + d + "/" + i + "/" + j + "/" + o + "/" + l + ".CSV/ch" + str(n+1))
        len_num = len(ch_list[n])
        x = np.linspace(0, len_num, len_num)
        y = ch_list[n]

        #フォント、ラベルの書き込み
        plt.rcParams['font.size'] = 15
        plt.title(str(n + 1) + 'ch')
        plt.xlabel('time')
        plt.ylabel('')

        plt.plot(x, y)
        plt.savefig(path.png + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "/ch" + str(n + 1))
        # plt.show()
        #4行2列の画像ができる
        #legendで凡例を消す
        #shareyでy軸の範囲を共通化(x軸を共通化するにはsharexを付け加える)
        if n == 6:
            df.plot(ylim=(-0.2,0.2),color="blue",subplots=True,layout=(4, 2),legend=False,
                    xticks=[0,64,128],fontsize=10)
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            plt.savefig(path.png + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "/mu"+l)
            # plt.show()
        plt.close('all')


def plot():
    # ファイルの読み込み
    for d in path.day:
        for i in path.subject:
            for j in path.char:
                for o in path.cut_time:
                    for l in path.time:
                        file_path = path.cut8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        df = pd.read_csv(file_path)

                        # print("test/" + d + "/" + i + "/" + j + "/data" + l + "cut.csv")
                        # print(df)

                        ch0 = list(df.iloc[:, 0].values)
                        ch1 = list(df.iloc[:, 1].values)
                        ch2 = list(df.iloc[:, 2].values)
                        ch3 = list(df.iloc[:, 3].values)
                        ch4 = list(df.iloc[:, 4].values)
                        ch5 = list(df.iloc[:, 5].values)
                        ch6 = list(df.iloc[:, 6].values)
                        # ch7 = list(df.iloc[:, 7].values)

                        # print("__________0ch__________")
                        # print(ch0)
                        # print("__________1ch__________")
                        # print(ch1)
                        # print("__________2ch__________")
                        # print(ch2)
                        # print("__________3ch__________")
                        # print(ch3)
                        # print("__________4ch__________")
                        # print(ch4)
                        # print("__________5ch__________")
                        # print(ch5)
                        # print("__________6ch__________")
                        # print(ch6)
                        # print("__________7ch__________")
                        # print(ch7)

                        # ch_list = [ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7]
                        ch_list = [ch0, ch1, ch2, ch3, ch4, ch5, ch6]

                        write_plot(ch_list, d, i, j, o, l, df)


if __name__ == '__main__':
    plot()
