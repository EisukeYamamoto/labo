"""
主成分分析
"""



import pandas as pd
from sklearn.decomposition import PCA
from app import path
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# 外れ値の関数
def outlier_iqr(df):

    for i in range(len(df.columns)):
        # 四分位数
        a = X_pca.iloc[:, i]

        Q1 = a.quantile(.30)
        Q3 = a.quantile(.70)
        # 四分位数
        q = Q3 - Q1

        # 外れ値の基準点
        outlier_min = Q1 - (q) * 1.5
        outlier_max = Q3 + (q) * 1.5

        # 範囲から外れている値を除く
        a[a < outlier_min] = None
        a[a > outlier_max] = None

    return df


train_x = []
train_y = []

# ファイルの読み込み
for d in path.test_day:
    for i in path.subject:
        for j in path.char:
            for l in path.time:
                file_path = path.cut + "/" + d + "/" + i + "/" + j + "/data" + l + "cut.csv"
                df = pd.read_csv(file_path)
                # print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")

                ch0 = list(df.iloc[:, 0].values)
                ch1 = list(df.iloc[:, 1].values)
                ch2 = list(df.iloc[:, 2].values)
                ch3 = list(df.iloc[:, 3].values)
                ch4 = list(df.iloc[:, 4].values)
                ch5 = list(df.iloc[:, 5].values)
                ch6 = list(df.iloc[:, 6].values)
                ch7 = list(df.iloc[:, 7].values)


                feature_value = []
                feature_value.extend(ch0)
                feature_value.extend(ch1)
                feature_value.extend(ch2)
                feature_value.extend(ch3)
                feature_value.extend(ch4)
                feature_value.extend(ch5)
                feature_value.extend(ch6)
                feature_value.extend(ch7)

                train_x.append(feature_value)
                # numpy.savetxt(path.pca + "/" + d + "/" + i + "/" + "data.csv", train_x, delimiter=',')

                if (j == 'あ'):
                    train_y.append(1)
                elif (j == 'い'):
                    train_y.append(2)
                elif (j == 'う'):
                    train_y.append(3)
                elif (j == 'え'):
                    train_y.append(4)
                elif (j == 'お'):
                    train_y.append(5)

                # np.savetxt(path.pca + "/" + d + "/" + i + "/" + "rabel.csv", train_y, delimiter=',')



#標準化
sc =StandardScaler()
x = sc.fit_transform(train_x)

#約8000次元から8次元に圧縮
#n_componentsで次元変更
pca = PCA(n_components=2)
#標準化したxデータをPCAで次元圧縮
X_pca = pd.DataFrame(pca.fit_transform(x))

#外れ値
z = outlier_iqr(X_pca)
# print(z)

#色の指定
colors = [plt.cm.nipy_spectral(i/4., 1) for i in range(4)]

plt.figure(figsize=(6, 6))
plt.scatter(z[:,0],z[:,1],alpha=0.8)
# for c,x_pca in zip(colors,z):
#     plt.scatter(X_pca.iloc[:,x_pca], X_pca.iloc[:,x_pca+1], alpha=0.8,c=train_y)
plt.grid()
#保存時ファイル名変更
# plt.savefig(path.pca+"/1day"+"/sub_A"+"/pca_plot")
plt.show()

print('各主成分の寄与率:', pca.explained_variance_ratio_)
print('寄与率の累積:', sum(pca.explained_variance_ratio_))
#
# print(pca.components_)
#
# #csvファイルに出力
# numpy.savetxt(path.pca + "/" + d + "/" + i + "/" + "pca.csv",pca.components_,delimiter=',')

