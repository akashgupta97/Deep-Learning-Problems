import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

path = r'C:/ass1/ass4/MovieDataset/'
train = pd.read_csv(path + "labeledTrainData.tsv", delimiter="\t")
test = pd.read_csv(path + "testData.tsv", delimiter="\t")

train.head(5)

