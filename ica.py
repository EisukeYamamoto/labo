import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from app import path
import matplotlib.pyplot as plt

def fast_ica_from_sklearn():
    df = pd.read_csv()

    data = df.drop()

    # 独立成分の数＝2
    decomposer = FastICA(n_components= 2)

    # データの平均を計算
    M = np.mean(data, axis=1)[:, np.newaxis]

    # 各データから平均を引く
    data2 = data-M

    # 平均0としたデータに対して、独立成分分析を実施
    decomposer.fit(data2)

    # 独立成分ベクトルを取得(D次元 x 独立成分数)
    S = decomposer.transform(data2)

    # 混合行列の計算（データ数 x 独立性分数）
    W = decomposer.mixing_

    # 混合行列と独立成分から元の信号dataを復元
    X = np.dot(S,W.T)
    X += M



    # 混合行列の擬似逆行列を取得
    W_inv = decomposer.components_

    plt.plot(W_inv)


if __name__ == '__main__':
    fast_ica_from_sklearn()