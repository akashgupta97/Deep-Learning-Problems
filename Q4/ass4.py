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


train_clean = train
test_clean = test

train_clean['review'] = train_clean['review'].apply(lambda x: clean_text(x))
test_clean['review'] = test_clean['review'].apply(lambda x: clean_text(x))

test_clean.head(5)

# Tokenize the reviews
all_reviews = list(train_clean['review']) + list(test_clean['review'])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("Fitting is complete.")

train_seq = tokenizer.texts_to_sequences(list(train_clean['review']))
print("train_seq is complete.")

test_seq = tokenizer.texts_to_sequences(list(test_clean['review']))
print("test_seq is complete")

max_review_length = 200

train_pad = pad_sequences(train_seq, maxlen=max_review_length)
print("train_pad is complete.")

test_pad = pad_sequences(test_seq, maxlen=max_review_length)
print("test_pad is complete.")

x_train, x_valid, y_train, y_valid = train_test_split(train_pad, train.sentiment, test_size=0.15, random_state=2)


def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


def get_test_batches(x, batch_size):
    '''Create the batches for the testing data'''
    n_batches = len(x) // batch_size
    x = x[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size]

def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, dropout, learning_rate, multiple_fc, fc_units):
    '''Build the Recurrent Neural Network'''
    tf.reset_default_graph()
    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        print("Input = ", inputs.shape)
    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
        print("labels = ", labels.shape)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # Create the embeddings
    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        print("embeddings = ", embedding.shape)
        embed = tf.nn.embedding_lookup(embedding, inputs)
        print("embed = ", embed.shape)
    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # print(lstm.output_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        # print(drop.output_shape)
        # cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        cell = drop
        # print(cell.shape)
