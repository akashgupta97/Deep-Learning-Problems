"""Example of creating a model. For an easier way, see example_wrappers.py"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorlm import Vocabulary, Dataset, GeneratingLSTM


TEXT_PATH = "C:/ass1/q3/tinytrain.txt"
DEV_PATH = "C:/ass1/q3/tinyvalid.txt"
BATCH_SIZE = 20
NUM_TIMESTEPS = 30

with tf.Session() as session:
    # Generate a vocabulary based on the text
    vocab = Vocabulary.create_from_text(TEXT_PATH, max_vocab_size=96, level="char")
    # Obtain input and target batches from the text file
    dataset = Dataset(TEXT_PATH, vocab, BATCH_SIZE, NUM_TIMESTEPS)
    # Create the model in a TensorFlow graph
    model = GeneratingLSTM(vocab_size=vocab.get_size(),
                           neurons_per_layer=100,
                           num_layers=2,
                           max_batch_size=BATCH_SIZE,
                           output_keep_prob=0.8)
    # Initialize all defined TF Variables
    session.run(tf.global_variables_initializer())
    # Do the training
    epoch = 1
    step = 1
    arr1 = []
    arr2 = []
    for epoch in range(101):
        for inputs, targets in dataset:
            loss = model.train_step(session, inputs, targets)
            step += 1
        if epoch % 10 == 0:
