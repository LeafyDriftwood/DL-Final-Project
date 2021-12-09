import datetime
import numpy as np
import tensorflow as tf
import transformers
from tqdm import tqdm


def argmax(vec):
    # return the argmax as a python int
    idx = tf.math.argmax(vec, axis=1)
    return idx


def pad_ts_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]
    timestamp = [item[3] for item in batch]
    print(len(target))


    data = tf.keras.preprocessing.sequence.pad_sequences(data, value=0)
    timestamp = tf.keras.preprocessing.sequence.pad_sequences(timestamp, value=0)

    target = tf.constant(target)
    tweet = tf.constant(tweet)

    return [target, tweet, data, timestamp]


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.datetime.timestamp(t))

    np.array(timestamp) - timestamp[-1]
    return timestamp

