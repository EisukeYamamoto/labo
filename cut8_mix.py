"""
データの切り出し(1024点 (8cut.ver)）
"""

from app import path
import pandas as pd
import csv

N = path.data_long

for d in path.mix_day:
    for i in path.mix_subject:
        for j in path.mix_char:
            for l in path.time:
                file_path = path.cut_mix + "/" + d + "/" + i + "/" + j + "/" + l + "cut.CSV"
                df = pd.read_csv(file_path)

                print("test/" + d + "/" + i + "/" + j + "/" + l + "cut.csv")

                csv_file = open(file_path, "r", encoding="ms932", errors="", newline="")
                f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)



                o2 = 0

                for o in path.cut8_time:
                    nf = open(path.cut8_mix + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV", 'w')
                    dataWriter = csv.writer(nf)



                    # データの切り出し

                    y1 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y2 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y3 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y4 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y5 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y6 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    y7 = df.iloc[int((N / 8) * o2):int((N / 8) * (o2+1)),:]
                    # y8 = df.iloc[int(128*o):int(128*(o+1)),:]
                    o2 += 1


                    # new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7, y8], axis=1)
                    new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7], axis=1)

                    # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                    new_data.columns = new_data.iloc[0, :]
                    new_data.index = new_data.iloc[:, 0]
                    # new_data = new_data.iloc[1:129, 1:8]
                    new_data = new_data.iloc[1:N + 1, 1:7]

                    new_data.to_csv(path.cut8_mix + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV")


                    nf.close()

