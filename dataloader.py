import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.utils import Sequence
import numpy as np
import keras


from utils import get_timestamp


class SuicidalDataset(Sequence):
    def __init__(self, label, tweet, temporal, timestamp, current=True, random=False):
        super().__init__()
        self.label = label
        self.tweet = tweet
        self.temporal = temporal
        self.current = current
        self.timestamp = timestamp
        self.random = random
        self.batch_size = 100

    def __len__(self):
        return len(self.label)

    # given an item, returns a list of labels, tweet features, temporal tweet features, and the timestamp
    def __getitem__(self, item):
        labels = (self.label[item])
        tweet_features = self.tweet[item]

        if self.current:
            result = self.temporal[item]
            if self.random:
                np.random.shuffle(result)
            
            temporal_tweet_features = tf.constant(result)
            timestamp = tf.constant(get_timestamp(self.timestamp[item]))
            
        else:
            if len(self.temporal[item]) == 1:
                temporal_tweet_features = np.zeros((1, 768), dtype=tf.float32)
                timestamp = np.zeros((1, 1), dtype=tf.float32)
                

            else:
                temporal_tweet_features = tf.constant(self.temporal[item][1:])
                timestamp = tf.constant(get_timestamp(self.timestamp[item][1:]))

        
        return [labels, tweet_features, temporal_tweet_features, timestamp]

