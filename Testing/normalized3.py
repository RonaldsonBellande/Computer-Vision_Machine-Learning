#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#loading data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data_gained = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

print(data_gained)

max_value = np.amax(data_gained)
min_value = np.amin(data_gained)
print("Max:",max_value,", Min:", min_value)

#Normalizing data by using ((data_gained - min_value)/ (max_value - min_value))
#Method 1
normalized1 = ((data_gained - min_value)/ (max_value - min_value))
print(normalized1)
plt.plot(normalized1)
plt.show()

#Normalized using normalize
#Method 2
from sklearn import preprocessing
print(data_gained.shape)
#reshaped it 
data_gained_reshaped = data_gained.reshape(1,-1)
normalize = preprocessing.normalize(data_gained_reshaped)
print(normalize)
#reshaped it back
print(normalize.shape)
#flatten it to plot it 
normalized2 = normalize.flatten()
print(normalized2.shape)
plt.plot(normalized2)
plt.show()

#Normalized Method 3
from sklearn.preprocessing import minmax_scale
normalized3 = minmax_scale(data_gained)
plt.plot(normalized3)
plt.show()

#Normalize method 4
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_gained_reshaped2 = data_gained.reshape(-1,1)
print(data_gained_reshaped2)
normalized4 = scaler.fit_transform(data_gained_reshaped2)
print(normalized4)
normalized4 = normalized4.flatten()
plt.plot(normalized4)
plt.show()


