# coding:utf-8

from matplotlib import pyplot as plt
from app import path
import pandas as pd
import numpy as np


def write_plot(ch_list, d, i, j, o, l, df):
    print("test/" + d + "/" + i + "/" + j + "/" + o + "/" + l + ".CSV/")
    len_num = len(ch_list[0])
    x = np.linspace(0, len_num, len_num)
    y0 = ch_list[0]
    y1 = ch_list[1]
    y2 = ch_list[2]
    y3 = ch_list[3]
    y4 = ch_list[4]
    y5 = ch_list[5]
    y6 = ch_list[6]

    labels0 = [-0.5,0.5]
    labels1 = [-0.5, 0.5]
    labels2 = [-0.5, 0.5]
    labels3 = [-0.5, 0.5]
    labels4 = [-0.5, 0.5]
    labels5 = [-0.5, 0.5]
    labels6 = [-0.5, 0.5]

    l0,l1,l2,l3,l4,l5,l6 = "ch0","ch1","ch2","ch3","ch4","ch5","ch6"
    o0,o1,o2,o3,o4,o5,o6 = 0,3,5,7,9,11,13

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
    yo0 = list(map(lambda x: x + o0, y0))
    yo1 = list(map(lambda x: x + o1, y1))
    yo2 = list(map(lambda x: x + o2, y2))
    yo3 = list(map(lambda x: x + o3, y3))
    yo4 = list(map(lambda x: x + o4, y4))
    yo5 = list(map(lambda x: x + o5, y5))
    yo6 = list(map(lambda x: x + o6, y6))


    plt.plot(x, yo6 , color="blue")
    plt.plot(x, yo5 , color="blue")
    plt.plot(x, yo4 , color="blue")
    plt.plot(x, yo3 , color="blue")
    plt.plot(x, yo2 , color="blue")
    plt.plot(x, yo1 , color="blue")
    plt.plot(x, yo0 , color="blue")

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
    plt.savefig(path.png_cnn_sub + "/" + i + "/" + d + "_" + j + "_" + o + "_" + l)
    #plt.show()
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
