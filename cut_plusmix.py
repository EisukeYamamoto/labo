"""
データの切り出し(1024点 (8cut.ver)）
"""

from app import path
import pandas as pd
import numpy as np
import csv

N = path.data_long
hammingWindow = np.hamming(256)

# 外れ値の関数
def outlier_iqr(df):

    for i in range(len(df.columns)):
        # 四分位数
        a = df.iloc[:, i]

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

    return df

for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.mix_char2:
            for l in path.time:
                file_path = path.cut_mix + "/" + d + "/" + i + "/" + j + "/" + l + "cut.CSV"
                df = pd.read_csv(file_path)
                df = outlier_iqr(df)

                print("test/" + d + "/" + i + "/" + j + "/" + l + "cut.csv")

                csv_file = open(file_path, "r", encoding="ms932", errors="", newline="")
                f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)



                o2 = 0

                for o in path.cutplusmix_time:
                    nf = open(path.cut_plusmix + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV", 'w')
                    dataWriter = csv.writer(nf)



                    # データの切り出し

                    y1 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y2 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y3 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y4 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    # y5 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    # y6 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    # y7 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    # y8 = df.iloc[int(128*o):int(128*(o+1)),:]
                    o2 += 0.5

                    # ハミング窓
                    y1 = y1.values
                    y2 = y2.values
                    y3 = y3.values
                    y4 = y4.values
                    # y5 = y5.values
                    # y6 = y6.values
                    # y7 = y7.values

                    y1 = hammingWindow * y1.T
                    y2 = hammingWindow * y2.T
                    y3 = hammingWindow * y3.T
                    y4 = hammingWindow * y4.T
                    # y5 = hammingWindow * y5.T
                    # y6 = hammingWindow * y6.T
                    # y7 = hammingWindow * y7.T

                    y1 = pd.DataFrame(y1.T)
                    y2 = pd.DataFrame(y2.T)
                    y3 = pd.DataFrame(y3.T)
                    y4 = pd.DataFrame(y4.T)
                    # y5 = pd.DataFrame(y5.T)
                    # y6 = pd.DataFrame(y6.T)
                    # y7 = pd.DataFrame(y7.T)


                    # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7, y8], axis=1)
                    # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7], axis=1)
                    new_data = pd.concat([ y1, y2, y3, y4], axis=1)

                    # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                    new_data.columns = new_data.iloc[0, :]
                    new_data.index = new_data.iloc[:, 0]
                    # new_data = new_data.iloc[1:129, 1:8]
                    # new_data = new_data.iloc[1:N + 1, 1:7]
                    new_data = new_data.iloc[ 1:N + 1, 1:4 ]

                    new_data.to_csv(path.cut_plusmix + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV")


                    nf.close()

