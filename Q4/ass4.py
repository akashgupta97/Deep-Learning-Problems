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

def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"   ", " ", text)  # Remove any extra spaces
    text = re.sub(r"  ", " ", text)
    # Return a list of words
    return (text)

