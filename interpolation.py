"""
線形補間
"""
from app import path
import pandas as pd
import numpy as np
from scipy import interpolate
import csv

for d in path.test_day:
    for i in path.subject:
        for j in path.char:
            for l in path.time:
                file_path = path.csv + "/" + d + "/" + i + "/" + j + "/data" + l + ".csv"
                df = pd.read_csv(file_path)

                csv_file = open(file_path, "r", encoding="ms932", errors="", newline="")
                f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"',
                               skipinitialspace=True)

                nf = open(path.inte + "/" + d + "/" + i + "/" + j + "/" + "data" + l + "change.csv", 'w')
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
                before_data_8 = [x[7] for x in rowlist]

                #線形補間
                t = np.linspace(0, len(df), len(df) + 1)
                tt = np.linspace(0, len(df), 3500)

                f1 = interpolate.interp1d(t, before_data_1)
                y1 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_2)
                y2 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_3)
                y3 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_4)
                y4 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_5)
                y5 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_6)
                y6 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_7)
                y7 = f1(tt)
                f1 = interpolate.interp1d(t, before_data_8)
                y8 = f1(tt)


                #行と列を変換
                pd_y1 = pd.DataFrame({'ch1': y1})
                pd_y2 = pd.DataFrame({'ch2': y2})
                pd_y3 = pd.DataFrame({'ch3': y3})
                pd_y4 = pd.DataFrame({'ch4': y4})
                pd_y5 = pd.DataFrame({'ch5': y5})
                pd_y6 = pd.DataFrame({'ch6': y6})
                pd_y7 = pd.DataFrame({'ch7': y7})
                pd_y8 = pd.DataFrame({'ch8': y8})

                new_data = pd.concat([pd_y1,pd_y2,pd_y3,pd_y4,pd_y5,pd_y6,pd_y7,pd_y8],axis = 1)

                # 下の2行は0列目の全ての要素を参照,(1列目から8列目)までを代入
                new_data.index = new_data.iloc[:, 0]
                new_data = new_data.iloc[:, 1:9]
                new_data.to_csv(path.inte + "/" + d + "/" + i + "/" + j + "/" + "data" + l + "change.csv")

                # rowList_interpolated = []
                # rowList_interpolated.append(y1)
                # rowList_interpolated.append(y2)
                # rowList_interpolated.append(y3)
                # rowList_interpolated.append(y4)
                # rowList_interpolated.append(y5)
                # rowList_interpolated.append(y6)
                # rowList_interpolated.append(y7)
                # rowList_interpolated.append(y8)

                # print(rowList_interpolated)

                # dataWriter.writerows(a)

                nf.close()
