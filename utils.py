import datetime
import numpy as np
import tensorflow as tf
import transformers  # prob wrong
#from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# done converting except for some unsure parts


def argmax(vec):
    # return the argmax as a python int
    idx = tf.math.argmax(vec, axis=1)
    return idx.item()  # not sure abt this

def pad_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]

    lens = [len(x) for x in data]

    data = tf.keras.preprocessing.sequence.pad_sequences(data, value=0)

    #     data = torch.tensor(data)
    target = tf.Tensor(target)
    tweet = tf.Tensor(tweet)
    lens = tf.Tensor(lens)

    return [target, tweet, data, lens]


def pad_ts_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]
    timestamp = [item[3] for item in batch]

    lens = [len(x) for x in data]

    data = tf.keras.preprocessing.sequence.pad_sequences(data, value=0)
    timestamp = tf.keras.preprocessing.sequence.pad_sequences(timestamp, value=0)

    #     data = torch.tensor(data)
    target = tf.Tensor(target)
    tweet = tf.Tensor(tweet)
    lens = tf.Tensor(lens)

    return [target, tweet, data, lens, timestamp]


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.datetime.timestamp(t))

    np.array(timestamp) - timestamp[-1]
    return timestamp
