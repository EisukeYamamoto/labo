# coding:utf-8
"""
svm
"""
from app import path
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

N = 127
dt = 0.01
fc = 20


def svm():
    test_x = []
    test_y = []
    train_x = []
    train_y = []

    cost_value_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    gamma_value_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    # テストファイルの読み込み
    for d in path.test_day:
        for i in path.subject:
            for j in path.char:
                for o in path.cut_time:
                    for l in path.test_time:
                        file_path = path.cut8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        df = pd.read_csv(file_path)

                        print("test/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                        # print(df)

                        ch0 = list(df.iloc[:, 0].values)
                        ch1 = list(df.iloc[:, 1].values)
                        ch2 = list(df.iloc[:, 2].values)
                        ch3 = list(df.iloc[:, 3].values)
                        ch4 = list(df.iloc[:, 4].values)
                        ch5 = list(df.iloc[:, 5].values)
                        ch6 = list(df.iloc[:, 6].values)
                        # ch7 = list(df.iloc[:, 7].values)

                        n_ch0 = np.array(ch0)
                        n_ch1 = np.array(ch1)
                        n_ch2 = np.array(ch2)
                        n_ch3 = np.array(ch3)
                        n_ch4 = np.array(ch4)
                        n_ch5 = np.array(ch5)
                        n_ch6 = np.array(ch6)

                        # フーリエ変換

                        f_ch0 = np.fft.fft(n_ch0)
                        f_ch1 = np.fft.fft(n_ch1)
                        f_ch2 = np.fft.fft(n_ch2)
                        f_ch3 = np.fft.fft(n_ch3)
                        f_ch4 = np.fft.fft(n_ch4)
                        f_ch5 = np.fft.fft(n_ch5)
                        f_ch6 = np.fft.fft(n_ch6)

                        fq = np.linspace(0, 1.0 / dt, N)

                        f_ch0[(fq > fc)] = 0
                        f_ch1[(fq > fc)] = 0
                        f_ch2[(fq > fc)] = 0
                        f_ch3[(fq > fc)] = 0
                        f_ch4[(fq > fc)] = 0
                        f_ch5[(fq > fc)] = 0
                        f_ch6[(fq > fc)] = 0

                        F_ch0_abs = np.abs(f_ch0)
                        F_ch1_abs = np.abs(f_ch1)
                        F_ch2_abs = np.abs(f_ch2)
                        F_ch3_abs = np.abs(f_ch3)
                        F_ch4_abs = np.abs(f_ch4)
                        F_ch5_abs = np.abs(f_ch5)
                        F_ch6_abs = np.abs(f_ch6)

                        F_ch0_abs_amp = F_ch0_abs / N * 2
                        F_ch1_abs_amp = F_ch1_abs / N * 2
                        F_ch2_abs_amp = F_ch2_abs / N * 2
                        F_ch3_abs_amp = F_ch3_abs / N * 2
                        F_ch4_abs_amp = F_ch4_abs / N * 2
                        F_ch5_abs_amp = F_ch5_abs / N * 2
                        F_ch6_abs_amp = F_ch6_abs / N * 2

                        F_ch0_abs_amp[0] = F_ch0_abs_amp[0] / 2
                        F_ch1_abs_amp[0] = F_ch1_abs_amp[0] / 2
                        F_ch2_abs_amp[0] = F_ch2_abs_amp[0] / 2
                        F_ch3_abs_amp[0] = F_ch3_abs_amp[0] / 2
                        F_ch4_abs_amp[0] = F_ch4_abs_amp[0] / 2
                        F_ch5_abs_amp[0] = F_ch5_abs_amp[0] / 2
                        F_ch6_abs_amp[0] = F_ch6_abs_amp[0] / 2

                        f_ch0_ifft = np.fft.ifft(f_ch0)
                        f_ch1_ifft = np.fft.ifft(f_ch1)
                        f_ch2_ifft = np.fft.ifft(f_ch2)
                        f_ch3_ifft = np.fft.ifft(f_ch3)
                        f_ch4_ifft = np.fft.ifft(f_ch4)
                        f_ch5_ifft = np.fft.ifft(f_ch5)
                        f_ch6_ifft = np.fft.ifft(f_ch6)

                        f_ch0_ifft_real = f_ch0_ifft.real * 2
                        f_ch1_ifft_real = f_ch1_ifft.real * 2
                        f_ch2_ifft_real = f_ch2_ifft.real * 2
                        f_ch3_ifft_real = f_ch3_ifft.real * 2
                        f_ch4_ifft_real = f_ch4_ifft.real * 2
                        f_ch5_ifft_real = f_ch5_ifft.real * 2
                        f_ch6_ifft_real = f_ch6_ifft.real * 2

                        feature_value = []
                        feature_value.extend(f_ch0_ifft_real)
                        feature_value.extend(f_ch1_ifft_real)
                        feature_value.extend(f_ch2_ifft_real)
                        feature_value.extend(f_ch3_ifft_real)
                        feature_value.extend(f_ch4_ifft_real)
                        feature_value.extend(f_ch5_ifft_real)
                        feature_value.extend(f_ch6_ifft_real)
                        # feature_value.extend(ch0)
                        # feature_value.extend(ch1)
                        # feature_value.extend(ch2)
                        # feature_value.extend(ch3)
                        # feature_value.extend(ch4)
                        # feature_value.extend(ch5)
                        # feature_value.extend(ch6)
                        # feature_value.extend(ch7)

                        test_x.append(feature_value)

                        if (j == '指'):
                            test_y.append('指')
                        elif (j == '手首'):
                            test_y.append('手首')


    # ファイルの読み込み
    for d in path.test_day:
        for i in path.subject:
            for j in path.char:
                for o in path.cut_time:
                    for l in path.train_time:
                        file_path = path.cut8 + "/" + d + "/" + i + "/" + j + "/" + o + "/" + l + "cut.CSV"
                        df = pd.read_csv(file_path)
                        print("train/" + d + "/" + i + "/" + j + "/data" + l + ".csv")
                        # print(df)

                        ch0 = list(df.iloc[:, 0].values)
                        ch1 = list(df.iloc[:, 1].values)
                        ch2 = list(df.iloc[:, 2].values)
                        ch3 = list(df.iloc[:, 3].values)
                        ch4 = list(df.iloc[:, 4].values)
                        ch5 = list(df.iloc[:, 5].values)
                        ch6 = list(df.iloc[:, 6].values)
                        # ch7 = list(df.iloc[:, 7].values)

                        n_ch0 = np.array(ch0)
                        n_ch1 = np.array(ch1)
                        n_ch2 = np.array(ch2)
                        n_ch3 = np.array(ch3)
                        n_ch4 = np.array(ch4)
                        n_ch5 = np.array(ch5)
                        n_ch6 = np.array(ch6)

                        # フーリエ変換

                        f_ch0 = np.fft.fft(n_ch0)
                        f_ch1 = np.fft.fft(n_ch1)
                        f_ch2 = np.fft.fft(n_ch2)
                        f_ch3 = np.fft.fft(n_ch3)
                        f_ch4 = np.fft.fft(n_ch4)
                        f_ch5 = np.fft.fft(n_ch5)
                        f_ch6 = np.fft.fft(n_ch6)

                        fq = np.linspace(0, 1.0 / dt, N)

                        f_ch0[(fq > fc)] = 0
                        f_ch1[(fq > fc)] = 0
                        f_ch2[(fq > fc)] = 0
                        f_ch3[(fq > fc)] = 0
                        f_ch4[(fq > fc)] = 0
                        f_ch5[(fq > fc)] = 0
                        f_ch6[(fq > fc)] = 0

                        F_ch0_abs = np.abs(f_ch0)
                        F_ch1_abs = np.abs(f_ch1)
                        F_ch2_abs = np.abs(f_ch2)
                        F_ch3_abs = np.abs(f_ch3)
                        F_ch4_abs = np.abs(f_ch4)
                        F_ch5_abs = np.abs(f_ch5)
                        F_ch6_abs = np.abs(f_ch6)

                        F_ch0_abs_amp = F_ch0_abs / N * 2
                        F_ch1_abs_amp = F_ch1_abs / N * 2
                        F_ch2_abs_amp = F_ch2_abs / N * 2
                        F_ch3_abs_amp = F_ch3_abs / N * 2
                        F_ch4_abs_amp = F_ch4_abs / N * 2
                        F_ch5_abs_amp = F_ch5_abs / N * 2
                        F_ch6_abs_amp = F_ch6_abs / N * 2

                        F_ch0_abs_amp[0] = F_ch0_abs_amp[0] / 2
                        F_ch1_abs_amp[0] = F_ch1_abs_amp[0] / 2
                        F_ch2_abs_amp[0] = F_ch2_abs_amp[0] / 2
                        F_ch3_abs_amp[0] = F_ch3_abs_amp[0] / 2
                        F_ch4_abs_amp[0] = F_ch4_abs_amp[0] / 2
                        F_ch5_abs_amp[0] = F_ch5_abs_amp[0] / 2
                        F_ch6_abs_amp[0] = F_ch6_abs_amp[0] / 2

                        f_ch0_ifft = np.fft.ifft(f_ch0)
                        f_ch1_ifft = np.fft.ifft(f_ch1)
                        f_ch2_ifft = np.fft.ifft(f_ch2)
                        f_ch3_ifft = np.fft.ifft(f_ch3)
                        f_ch4_ifft = np.fft.ifft(f_ch4)
                        f_ch5_ifft = np.fft.ifft(f_ch5)
                        f_ch6_ifft = np.fft.ifft(f_ch6)

                        f_ch0_ifft_real = f_ch0_ifft.real * 2
                        f_ch1_ifft_real = f_ch1_ifft.real * 2
                        f_ch2_ifft_real = f_ch2_ifft.real * 2
                        f_ch3_ifft_real = f_ch3_ifft.real * 2
                        f_ch4_ifft_real = f_ch4_ifft.real * 2
                        f_ch5_ifft_real = f_ch5_ifft.real * 2
                        f_ch6_ifft_real = f_ch6_ifft.real * 2

                        feature_value = []
                        feature_value.extend(f_ch0_ifft_real)
                        feature_value.extend(f_ch1_ifft_real)
                        feature_value.extend(f_ch2_ifft_real)
                        feature_value.extend(f_ch3_ifft_real)
                        feature_value.extend(f_ch4_ifft_real)
                        feature_value.extend(f_ch5_ifft_real)
                        feature_value.extend(f_ch6_ifft_real)
                        # feature_value.extend(ch0)
                        # feature_value.extend(ch1)
                        # feature_value.extend(ch2)
                        # feature_value.extend(ch3)
                        # feature_value.extend(ch4)
                        # feature_value.extend(ch5)
                        # feature_value.extend(ch6)
                        # feature_value.extend(ch7)

                        train_x.append(feature_value)

                        if (j == '指'):
                            train_y.append('指')
                        elif (j == '手首'):
                            train_y.append('手首')


    train_xn = train_x.astype('float32')
    train_x = train_xn / 255.0



    print(train_x)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_x.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size))
    model.add(Activation('softmax'))

    model.summary()
    optimizers = "Adadelta"
    results = {}

    epochs = 50
    model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
    results[0] = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs)

    model_json_str = model.to_json()
    open('mnist_mlp_model.json', 'w').write(model_json_str)
    model.save_weights('mnist_mlp_weights.h5');

    score = model.evaluate(X_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    x = range(epochs)
    for k, result in results.items():
        plt.plot(x, result.history['acc'], label=k)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, ncol=2)

    name = 'acc.jpg'
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    for k, result in results.items():
        plt.plot(x, result.history['val_acc'], label=k)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, ncol=2)

    name = 'val_acc.jpg'
    plt.savefig(name, bbox_inches='tight')


if __name__ == '__main__':
    svm()
