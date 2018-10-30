"""Example of creating a model. For an easier way, see example_wrappers.py"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorlm import Vocabulary, Dataset, GeneratingLSTM


TEXT_PATH = "C:/ass1/q3/tinytrain.txt"
DEV_PATH = "C:/ass1/q3/tinyvalid.txt"
BATCH_SIZE = 20
NUM_TIMESTEPS = 30
