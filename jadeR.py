#!/usr/bin/env python

#  https://github.com/camilleanne/pulse/blob/master/jade.py 参照
#######################################################################
# jadeR.py -- Blind source separation of real signals
#
# Version 1.8
#
# Copyright 2005, Jean-Francois Cardoso (Original MATLAB code)
# Copyright 2007, Gabriel J.L. Beckers (NumPy translation)
#
# このプログラムはフリーソフトウェアです。
# あなたは、フリーソフトウェアファウンデーションによって公開されている
# GNU一般公衆利用許諾書のバージョン3、またはそれ以降のバージョンのいずれかの条項の下で、
# これを再配布したり、変更したりすることができます(あなたの選択で)。
#
# このプログラムは有用であることを期待して配布されていますが、商品性や特定目的への適合性
# についての暗黙の保証もなく、何らの保証もありません。
# 詳細はGNU一般公衆利用許諾書を参照してください。
#
# このプログラムと一緒にGNU一般公衆利用許諾書のコピーを受け取っているはずです。
# そうでない場合は、<http://www.gnu.org/licenses/>を参照してください。
#######################################################################

# This file can be either used from the command line (type
# 'python jadeR.py --help' for usage, or see docstring of function main below)
# or it can be imported as a module in a python shell or program (use
# 'import jadeR').

# Comments in this source file are from the original MATLAB program, unless they
# are preceded by 'GB'.


"""
jadeR
このモジュールには、実信号のブラインドソース分離を行う jadeR という関数が 1 つだけ含まれています。
願わくば、より多くのICAアルゴリズムが将来的に追加されることを期待しています。
jadeRにはNumPyが必要です。
"""

from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float32, float64, loadtxt, matrix, multiply, ndarray, \
    newaxis, savetxt, sign, sin, sqrt, zeros
from numpy.linalg import eig, pinv


def jadeR(X):
    """
    JADEによる実信号のブラインド分離
    jadeRは独立成分分析（ICA）アルゴリズムであるJADEを実装しています。
    ジャン・フランコワ・カルドソが開発しました。
    JADEについての詳細な情報は、以下のサイトを参照してください。
    Cardoso, J. (1999) High-order contrasts for independent component analysis. Neural Computation, 11(1): 157-192.
    または、ウェブサイトをご覧ください: http://www.tsi.enst.fr/~cardoso/guidesepsou.html

    ICAに関するより詳しい情報は、Hyvarinen A., Karhunen J., Oja E. (2001). Independent Component Analysis, Wiley.
    またはウェブサイト http://www.cis.hut.fi/aapo/papers/IJCNN99_tutorialweb/
    オリジナルのMatlabバージョン1.8 (2005年5月)からNumPyへの翻訳
    ガブリエル・ベッカーズ、http://gbeckers.nl .

    Parameters:
        X -- n x T のデータ行列（n個のセンサ，T個のサンプル）．NumPy 配列または行列でなければなりません．
        m -- 抽出する独立成分の数．出力行列Bは、m個のソースのみが抽出されるように、サイズm×nを持つことになります。
             これは，jadeR の操作を m 個の第一主成分に制限することで行われます．
             デフォルトは None で、この場合は m == n です。
        verbose -- 進捗情報を表示します。デフォルトは False です。

    Returns:
        Y＝B＊Xがｎ＊Tのデータ行列Xから抽出された分離されたソースであるようなｍ＊ｎの行列B（Numpy行列型）。
        Bの行は、pinv(B)の列がノルムが小さい順になるように順序づけられます。
        これは、`最もエネルギー的に有意な`成分がY = B * Xの行に最初に現れるという効果があります。

    Quick notes (more at the end of this file):
    o このコードは、REAL-valuedシグナルのためのコードです。 実数および複素数信号用のJADEのMATLAB実装も
       http://sig.enst.fr/~cardoso/stuff.html から入手可能です。
    o このアルゴリズムは、より効率的に処理するように最適化されているという点で、
      最初にリリースされたJADEの実装とは異なります。
        1) 実信号を用いて
        2) ICAモデルが必ずしも保持されていない場合を想定しています。
    o この実装で抽出できる独立したコンポーネントの数には実用的な制限があります。
      JADEの最初のステップは、次元数をnからm（デフォルトはn）に削減したPCAであることに注意してください。
      実際には、mは`非常に大きくはできません(40, 50, 60以上...利用可能なメモリに依存します)
    o このファイルの最後にあるメモ、リファレンス、リビジョンの履歴など、WEB上の詳細を参照してください。
        http://sig.enst.fr/~cardoso/stuff.html
    o NumPy翻訳の詳細については、このファイルの最後を参照してください。
    o このコードは良い仕事をしているはずなのに!  NumPYコードに関する問題があれば報告してください
     gabriel@gbeckers.nl
    著作権はオリジナルのMatlabコードにあります。Jean-Francois Cardoso <cardoso@sig.enst.fr>
    著作権Numpy翻訳。ガブリエル・ベッカーズ <gabriel@gbeckers.nl
    """

    # GB: 私たちは入力引数のチェックを行い、
    # 元の入力に干渉しないように新しい変数にデータをコピーします。
    # また、倍精度 (float64) と X 用の numpy 行列型も必要です。

    origtype = X.dtype  # float64

    X = matrix(X.astype(float64))  # 浮動小数点64の配列として作成されたXのコピーから行列を作成します。

    [n, T] = X.shape

    m = n

    X -= X.mean(1)

    # 白色化と信号部分空間への投影
    # -------------------------------------------

    # 標本共分散行列の固有基底
    [D, U] = eig((X * X.T) / float(T))
    # バリアンスの増加で並べ替え
    k = D.argsort()
    Ds = D[k]

    # 分散の減少によるmの最も有意なプリンシパルの比較
    PCs = arange(n - 1, n - m - 1, -1)

    # PCA
    # この段階で、Bはm個の成分についてPCAを行います。
    B = U[:, k[PCs]].T

    # --- Scaling ---------------------------------
    # 主成分のスケール
    scales = sqrt(Ds[ PCs ])
    B = diag(1. / scales) * B
    # Sphering
    X = B * X

    # 簡単なところをやってみました。Bは白化行列で、Xは白です。

    del U, D, Ds, k, PCs, scales

    # NOTE: この段階では、X は、すべてのエントリが単位分散を持つようになったことを除いて、
    # 実データの m 個の成分での PCA 分析です。Xをさらに回転させても、
    # Xが無相関成分のベクトルであるという性質は維持されます。X のエントリが無相関であるだけでなく、
    # 「可能な限り独立している」ような回転行列を見つけることが残っています。
    # この独立性は、2よりも高い次数の相関によって測定されます。 我々はこのような独立性の尺度を定義しているが、
    # これは 1) 相互情報の合理的な近似である 2) 高速アルゴリズムによって最適化できる
    # この独立性の尺度は、キュムラント行列の集合の「双対性」にも対応する。以下のコードは、
    # 積行列の特定の集合を最も対角化する行列として `missing rotation " を求めます。

    # 累積行列の推定
    # -------------------------------

    # データの再整形、少しでも速くなることを願って...
    X = X.T  # データを(256, 3)に転置します。
    # 実数対称行列の空間の Dim.
    dimsymm = (m * (m + 1)) / 2  # 6
    # 累積行列数
    nbcm = dimsymm  # 6
    # 積算行列の格納
    CM = matrix(zeros([m, m * nbcm], dtype=float64))
    R = matrix(eye(m, dtype=float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]
    # 積算行列の Temp.
    Qij = matrix(zeros([m, m], dtype=float64))
    # Temp
    Xim = zeros(m, dtype=float64)
    # Temp
    Xijm = zeros(m, dtype=float64)

    # シンメトリーの仕掛けを使ってストレージを節約しています。
    # 私はここで何が起こっているのかを説明する短いメモを書く必要があります。
    Range = arange(m)  # [0 1 2]

    for im in range(m):
        Xim = X[:, im]
        Xijm = multiply(Xim, Xim)
        Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * dot(R[:, im], R[:, im].T)
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:, jm])
            Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T
            CM[:, Range] = Qij
            Range = Range + m

    # これで nbcm = m(m+1)/2 キュムラント行列が大きな行列に格納されました。
    # m x m*nbcmの配列です。

    # 積算行列の合同対角化
    # ==============================================

    V = matrix(eye(m, dtype=float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]

    Diag = zeros(m, dtype=float64)  # [0. 0. 0.]
    On = 0.0
    Range = arange(m)  # [0 1 2]
    for im in range(nbcm):  # nbcm == 6
        Diag = diag(CM[:, Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM, CM).sum(axis=0)).sum(axis=0) - On
    # 小さな "角度の統計的にスケーリングされたしきい値
    seuil = 1.0e-6 / sqrt(T)  # 6.25e-08
    # sweep number
    encore = True
    sweep = 0
    # 総回転数
    updates = 0
    # 所定の深度での回転数
    upds = 0
    g = zeros([2, nbcm], dtype=float64)  # [[ 0.  0.  0.  0.  0.  0.] [ 0.  0.  0.  0.  0.  0.]]
    gg = zeros([2, 2], dtype=float64)  # [[ 0.  0.]  [ 0.  0.]]
    G = zeros([2, 2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # ジョイントの対角化適正化

    while encore:
        encore = False
        sweep = sweep + 1
        upds = 0
        Vkeep = V

        for p in range(m - 1):  # m == 3
            for q in range(p + 1, m):  # p == 1 | range(p+1, m) == [2]

                Ip = arange(p, m * nbcm, m)  # [ 0  3  6  9 12 15] [ 0  3  6  9 12 15] [ 1  4  7 10 13 16]
                Iq = arange(q, m * nbcm, m)  # [ 1  4  7 10 13 16] [ 2  5  8 11 14 17] [ 2  5  8 11 14 17]

                # computation of Givens angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]  # -6.54012319852 4.44880758012 -1.96674621935
                toff = gg[0, 1] + gg[1, 0]  # -15.629032394 -4.3847687273 6.72969915184
                theta = 0.5 * arctan2(toff, ton + sqrt(
                    ton * ton + toff * toff))  # -0.491778606993 -0.194537202087 0.463781701868
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0  # 5.87059352069 0.449409565866 2.24448683877

                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s], [s, c]])  # DON"T PRINT THIS! IT"LL BREAK THINGS! HELLA LONG
                    pair = array([p, q])  # don't print this either
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, concatenate([Ip, Iq])] = append(c * CM[:, Ip] + s * CM[:, Iq],
                                                              - s * CM[:, Ip] + c * CM[:, Iq], axis=1)
                    On = On + Gain
                    Off = Off - Gain
        updates = updates + upds  # 3 6 9 9

    # 分離行列
    # -------------------

    B = V.T * B  # [[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008]  [ 1.12505903 -2.42824508  0.92226197]]

    # 分離行列 B の行をパーミュレートして、最初に最もエネルギーの高い成分を取得します。ここでは**シグナル**は単位分散に正規化されています.
    # したがって，ソートは，A = pinv(B) の列のノルムに従って行われます．

    A = pinv(B)  # [[-3.35031851 -2.14563715  0.60277625] [-2.49989794 -1.25230985 -0.0835184 ] [-2.49501641 -0.67979249  0.12907178]]
    keys = array(argsort(multiply(A, A).sum(axis=0)[ 0 ]))[ 0 ]  # [2 1 0]
    B = B[keys, :]  # [[ 1.12505903 -2.42824508  0.92226197] [-0.41923305 -0.84589716  1.41050008] [ 0.17242566  0.10485568 -0.7373937 ]]
    B = B[::-1, :]  # [[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
    # just a trick to deal with sign == 0
    b = B[:, 0]  # [[ 0.17242566] [-0.41923305] [ 1.12505903]]
    signs = array(sign(sign(b) + 0.1).T)[ 0 ]  # [1. -1. 1.]
    B = diag(signs) * B  # [[ 0.17242566  0.10485568 -0.7373937 ] [ 0.41923305  0.84589716 -1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
    return B


# MATLABコードの修正履歴:

# - V1.8, December 2013
#  - modifications to main function for demo by Camille Teicheira
#  - also added inline comments of expected outputs for demo data
#  - demo here: http://github.com/camilleanne/pulsation
#
# - V1.8, May 2005
#  - Added some commented code to explain the cumulant computation tricks.
#  - Added reference to the Neural Comp. paper.
#
# -  V1.7, Nov. 16, 2002
#   - Reverted the mean removal code to an earlier version (not using
#     repmat) to keep the code octave-compatible.  Now less efficient,
#     but does not make any significant difference wrt the total
#     computing cost.
#   - Remove some cruft (some debugging figures were created.  What
#     was this stuff doing there???)
#
#
# -  V1.6, Feb. 24, 1997
#   - Mean removal is better implemented.
#   - Transposing X before computing the cumulants: small speed-up
#   - Still more comments to emphasize the relationship to PCA
#
#   V1.5, Dec. 24 1997
#   - The sign of each row of B is determined by letting the first element
#     be positive.
#
# -  V1.4, Dec. 23 1997
#   - Minor clean up.
#   - Added a verbose switch
#   - Added the sorting of the rows of B in order to fix in some reasonable
#     way the permutation indetermination.  See note 2) below.
#
# -  V1.3, Nov.  2 1997
#   - Some clean up.  Released in the public domain.
#
# -  V1.2, Oct.  5 1997
#   - Changed random picking of the cumulant matrix used for initialization
#     to a deterministic choice.  This is not because of a better rationale
#     but to make the ouput (almost surely) deterministic.
#   - Rewrote the joint diag. to take more advantage of Matlab"s tricks.
#   - Created more dummy variables to combat Matlab"s loose memory
#     management.
#
# -  V1.1, Oct. 29 1997.
#    Made the estimation of the cumulant matrices more regular. This also
#    corrects a buglet...
#
# -  V1.0, Sept. 9 1997. Created.
#
# Main references:
# @article{CS-iee-94,
#  title 	= "Blind beamforming for non {G}aussian signals",
#  author       = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
#  HTML 	= "ftp://sig.enst.fr/pub/jfc/Papers/iee.ps.gz",
#  journal      = "IEE Proceedings-F",
#  month = dec, number = 6, pages = {362-370}, volume = 140, year = 1993}
#
#
# @article{JADE:NC,
#  author  = "Jean-Fran\c{c}ois Cardoso",
#  journal = "Neural Computation",
#  title   = "High-order contrasts for independent component analysis",
#  HTML    = "http://www.tsi.enst.fr/~cardoso/Papers.PS/neuralcomp_2ppf.ps",
#  year    = 1999, month = jan, volume = 11, number = 1, pages = "157-192"}
#
#
#  Notes:
#  ======
#
#  Note 1) オリジナルのJadeアルゴリズム/コードは、ガウスノイズ白の複雑な信号を扱い、
#  独立成分のモデルが実際に保持されているという仮定を利用しています。
#  これは、いくつかの狭帯域の信号を扱う場合には、妥当な仮定です。
#  この文脈では、1つは、i)ホワイトニング過程でのノイズを正確に扱うことを真剣に検討し、
#  ii)4次の情報を効率的にまとめるために、少数の有意な固有行列を使用することを期待します。
# これらはすべてJADEアルゴリズムで行われます。
#
#  この実装では、実値信号を扱い、ICAモデルが正確に保持されることを期待していません。
#  したがって、加法ノイズを正確に扱おうとすることは無意味であり、
#  キュムラント・テンソルが最初のn個の固有行列で正確に要約できる可能性は非常に低いと考えられます。
#  そこで、固有行列の*全体*集合の合同対角化を考慮します。しかし、このような場合には、
#  積乱テンソルの「平行スライス」を等価的に用いることができるので、固有行列を計算する必要はありません。
#  計算のこの部分（固有行列の計算）は節約できます：積乱行列の集合を共同で対角化すれば十分です。
#  また、我々は実数信号を扱っているので、キュムラントの対称性を利用して対角化する行列の数を
#  さらに減らすことが容易になります。これらの考慮事項は、他の安価なトリックと一緒に、実数の混合物を扱い、
#  「モデルの外」で動作するように最適化されたJADEのこのバージョンにつながります。オリジナルのJADEとして
#  アルゴリズムは、キュムラントの「良い集合」を最小化することで動作します。
#
#  Note 2) 分離行列Bの行は、対応する混合行列A=pinv(B)の列が(ユークリッド)ノルムの降順になるように
#  ソートされています。これは，順列の不確定性を固定するための単純な，`ほぼ正統的な`方法である．
#  これは、回収された信号の最初の行(すなわち、B*Xの最初の行)が最もエネルギッシュな*成分*に対応する
#  という効果があります。しかし、S=B*Xのソース信号は単位分散を持っていることを思い出してください。
#  したがって、オブザベーションがエネルギーが小さい順に非混合であると言うとき、このエネルギッシュな
#  シグネチャは A=pinv(B)であり、分離されたソース信号の分散としてではない。
#
#  Note 3) JADEがB=jadeR(X,m)として実行され、mが値の範囲で変化する実験では、
#  分解の安定性をテストできるのは良いことです。このようなテストを支援するために、
#  Bの行を上述のように並べ替えることができます。また、各行の符号は、任意ではありますが
#  固定された方法で固定することにしました。慣例は、Bの各行の最初の要素が正であることです。
#
#  Note 4) 他の多くのICAアルゴリズムとは異なり、JADE（または少なくともこのバージョン）は
#  データそのものではなく、統計量（4次キュムラントの完全集合）に基づいて動作します。
#  これは以下の行列CMで表され、そのサイズはm^2 x m^2として大きくなります。
# 結果として、(このバージョンの)JADEは、おそらく「大規模」な数のソースで窒息することになるでしょう。
# ここでの「大量」というのは、主に利用可能なメモリに依存しており、40とかそこらのものになるかもしれません。
# 近いうちに、私は `statistic' オプションではなく `data' オプションを取るバージョンの JADE を準備するつもりです。

# Notes on translation (GB):
# =========================
#
# Note 1) 関数 jadeR は、オリジナルの MATLAB コードからの比較的リテラルな翻訳です。
# NumPy用に最適化することについては本当に調べたことがありません。これを見て良いアイデアがあれば教えてください。
#
# Note 2) NumPy出力とオリジナルMATLABスクリプトのOctave（MATLABクローン）
# 出力を比較するテストモジュールが利用可能です。


def main(X):
    B = jadeR(X)
    Y = B * matrix(X)
    print(Y.T)
    return Y.T

# B = B.astype(origtype)
# savetxt("ct_jade_data.txt", Y.T)