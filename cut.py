"""
データの切り出し(1024点）
"""

from app import path
import pandas as pd
import csv

N = path.data_long


for d in path.day:
    for i in path.subject:
        for j in path.char:
            for l in path.time:
                file_path = path.csv + "/" + d + "/" + i + "/" + j + "/" + l + ".CSV"
                df = pd.read_csv(file_path)

                print("test/" + d + "/" + i + "/" + j + "/" + l + ".csv")

                csv_file = open(file_path, "r", encoding="ms932", errors="", newline="")
                f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)

                nf = open(path.cut + "/" + d + "/" + i + "/" + j + "/" + l + "cut.CSV", 'w')
                dataWriter = csv.writer(nf)

                rowlist = []

                rowlist.append(next(f))
                for row in f:
                    rowlist.append(row)

                before_data_1 = [x[0] for x in rowlist]
                before_data_2 = [x[1] for x in rowlist]
                before_data_3 = [x[2] for x in rowlist]
                before_data_4 = [x[3] for x in rowlist]
                before_data_5 = [x[4] for x in rowlist]
                before_data_6 = [x[5] for x in rowlist]
                before_data_7 = [x[6] for x in rowlist]
                # before_data_8 = [x[7] for x in rowlist]

                #データの切り出し
                s_length1 = len(before_data_1) * 2 / 3
                s_length2 = len(before_data_2) * 2 / 3
                s_length3 = len(before_data_3) * 2 / 3
                s_length4 = len(before_data_4) * 2 / 3
                s_length5 = len(before_data_5) * 2 / 3
                s_length6 = len(before_data_6) * 2 / 3
                s_length7 = len(before_data_7) * 2 / 3
                # s_length8 = len(before_data_8)/ 2


                y1 = df.iloc[int(s_length1 - N/2):int(s_length1 + N/2 + 1),:]
                y2 = df.iloc[int(s_length2 - N/2):int(s_length2 + N/2 + 1),:]
                y3 = df.iloc[int(s_length3 - N/2):int(s_length3 + N/2 + 1),:]
                y4 = df.iloc[int(s_length4 - N/2):int(s_length4 + N/2 + 1),:]
                y5 = df.iloc[int(s_length5 - N/2):int(s_length5 + N/2 + 1),:]
                y6 = df.iloc[int(s_length6 - N/2):int(s_length6 + N/2 + 1),:]
                y7 = df.iloc[int(s_length7 - N/2):int(s_length7 + N/2 + 1),:]
                # y8 = df.iloc[int(s_length8 - 512):int(s_length8 + 513),:]
                print(y1)


                # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7, y8], axis=1)
                new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7], axis=1)

                # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                new_data.columns = new_data.iloc[0,:]
                new_data.index = new_data.iloc[:, 0]
                # new_data = new_data.iloc[1:1026, 1:8]
                new_data = new_data.iloc[1:N + 2, 1:7]
                # print(new_data.shape)

                new_data.to_csv(path.cut + "/" + d + "/" + i + "/" + j + "/" + l + "cut.csv")


                nf.close()

