# Separation of Compound Actions with Wrist and Finger Based on EMG
This repository is a backup of the source code used in my research. My research is in the development stage and I will keep updating it.\\
※Please note that we cannot release the experimental data used in this study due to portrait rights.

## Abstract
In this research, we propose to measure the EMGs of the wrist and fingers using dry-type sensors worn near the wrist, and to separate the measured data into wrist and finger EMGs by using independent component analysis (ICA). Then we can confirm the EMGs of the wrist and fingers from the complex motion and realize individual identification in more complex motions. The final goal of this study is to identify individual motions from complex motions. In this paper, as a preliminary step, the ICA is used to isolate compound motions and the validity of the method is evaluated. We had three days and four movements were measured. The results of the combination of FastICA, Infomax and JADE, respectively, were evaluated by the correlation coefficient with the original signal. The most accurate combination was FastICA + Infomax with an accuracy of 70.5%.
![MyResearch_Image](https://user-images.githubusercontent.com/51312413/95956885-e8683400-0e39-11eb-910f-f53b53e34a57.png)

## What is EMG?
EMG(ElectroMyoGraphy) is a biological signal and refers to an electrical signal that flows from the brain to the muscle fibers(Strictly speaking, it is a time-series signal of the potential change caused by membrane currents flowing through the volume conductors around the muscle fibers.). EMGs are often used to control prosthetic limbs and robots, and are also used for personal identification, as they can be differentiated by the amount of force applied and other factors.
![EMG_Image](https://user-images.githubusercontent.com/51312413/95958678-70e7d400-0e3c-11eb-9be2-495cf8a6bc21.png)

## 
