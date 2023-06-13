import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
import visualkeras
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

# loading in datasets
imdb = pd.read_csv('B3FD_metadata/B3FD-IMDB_age_gender.csv', delimiter=' ')
wiki = pd.read_csv('B3FD_metadata/B3FD-WIKI_age_gender.csv', delimiter=' ')

# combining datatsets
df = pd.concat([imdb, wiki])
df = df.reset_index(drop=True)

# function to group ages
def age_split(age):
    if age < 18:
        return 'X'
    if age in range(18,25):
        return 'A'
    if age in range(25,35):
        return 'B'
    if age in range(35,45):
        return 'C'
    if age in range(45,55):
        return 'D'
    if age in range(55,65):
        return 'E'
    else:
        return 'F'
    
df['age'] = df['age'].apply(age_split)

# Dropping indiviudals under 18
df = df.drop(df.loc[df['age'] == 'X'].index)
df = df.reset_index(drop=True)

ohe = OneHotEncoder(sparse=False)
age = ohe.fit_transform(df[['age']]).tolist()
df['age'] = age
df['gender'] = df['gender'].str.get_dummies()['M']
df.head(10)