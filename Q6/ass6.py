import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

K.clear_session()

np.random.seed(237)

train_orig = pd.read_csv('C:/ass1/train.csv')
test_orig = pd.read_csv('C:/ass1/test.csv')

train_orig.head()
