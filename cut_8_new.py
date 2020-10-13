"""
データの切り出し(2048点）
"""

from app import path
import pandas as pd
import numpy as np
import csv

N = path.data_long
hammingWindow = np.hamming(N+1)

for i in path.new_subject:
    for d in path.new_day:
        for s in path.sets:
            for j in path.mix_char2:
                for l in path.time:
                    file_path = path.cut_mix_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + l + "cut.CSV"
                    df = pd.read_csv(file_path)

                    print("newdata/" + i + "/" + d + "/" + s + "/" + j + "/" + l + ".csv")

                    csv_file = open(file_path, "r", encoding="ms932", errors="", newline="")
                    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)

                    o2 = 0

                    for o in path.cut8_time:
                        nf = open(path.cut_8_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "cut.CSV", 'w')
                        dataWriter = csv.writer(nf)

                        # データの切り出し

                        y1 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y2 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y3 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y4 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y5 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y6 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y7 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        y8 = df.iloc[ int((N / 8) * o2):int((N / 8) * (o2 + 1)), : ]
                        o2 += 1

                        # y1 = np.abs(y1)
                        # y2 = np.abs(y2)
                        # y3 = np.abs(y3)
                        # y4 = np.abs(y4)
                        # y5 = np.abs(y5)
                        # y6 = np.abs(y6)
                        # y7 = np.abs(y7)
                        # y8 = np.abs(y8)

                        new_data = pd.concat([y1, y2, y3, y4, y5, y6, y7, y8], axis=1)

                        # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                        new_data.columns = new_data.iloc[ 0, : ]
                        new_data.index = new_data.iloc[ :, 0 ]
                        new_data = new_data.iloc[ 1:N + 1, 1:8 ]

                        new_data.to_csv(path.cut_8_new + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "cut.CSV")
                        print("Cleate/" + "/" + i + "/" + d + "/" + s + "/" + j + "/" + o + "/" + l + "cut.CSV")

                        nf.close()

