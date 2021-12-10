import argparse
import copy
import json
import os
import pickle
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from dataloader import SuicidalDataset
from model.model import HistoricCurrent, Historic, Current
from utils import pad_ts_collate


# Tensorflow equivalent imports
import tensorflow as tf
import tensorflow.keras.backend as K

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    # Calculate binary crossentropy loss
    BCLoss_layer = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction= tf.losses.Reduction.NONE)
    temp_labels = tf.expand_dims(labels, 2)
    temp_logits = tf.expand_dims(logits, 2)
    BCLoss = BCLoss_layer(temp_labels, temp_logits)
    
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = tf.math.exp(-gamma * labels * logits - gamma * tf.math.log(1 + tf.math.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = tf.reduce_sum(weighted_loss)

    focal_loss /= tf.reduce_sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = tf.cast(tf.one_hot(labels, no_of_classes), tf.float32)

    weights = tf.constant(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, 0)
    weights = tf.repeat(weights, repeats=labels_one_hot.shape[0], axis=0)
    weights = tf.repeat(weights, repeats=1, axis=1) * labels_one_hot
    weights = tf.reduce_sum(weights, axis=1)
    
    weights = tf.expand_dims(weights, 1)
    weights = tf.repeat(weights, repeats=1, axis = 0)
    weights = tf.repeat(weights, repeats=no_of_classes, axis=1)
   
    # Return loss based on which type of loss was passed

    # Focal attempts to handle class impalances
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    
    elif loss_type == "sigmoid":
        cb_loss = tf.keras.metrics.binary_crossentropy(labels_one_hot, logits, from_logits =True)
        cb_loss = K.mean(cb_loss*weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = tf.keras.metrics.binary_crossentropy(labels_one_hot, logits)
        cb_loss = K.mean(cb_loss*weights)

    return cb_loss


def train_loop(model, dataloader, optimizer, device, dataset_len):
    # Train loop for data
    dataset_len = 80
    running_loss = 0.0
    running_corrects = 0
    
    new_arr = []

    for i in range(80):
      new_arr.append(dataloader.__getitem__(i))
    
    # Extracts relevant portions in train's inputs
    labels, tweet_features, temporal_features, timestamp = pad_ts_collate(new_arr)


    with tf.GradientTape() as tape:
        output = model.forward(tweet_features, temporal_features, timestamp)
        # Max value of all elements in output
      
        preds = tf.math.argmax(output, 1)
       


        unique_counts = [len([x for x in np.asarray(labels) if x == 0]), len([x for x in np.asarray(labels) if x == 1])]
        # Calculate loss
        loss = loss_fn(output, labels, unique_counts)
    # Calculates derivative of loss with respect to every tensor that can be trained
    gradients = tape.gradient(loss, model.trainable_variables)
    # Apply changes in gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Sum loss
    running_loss += loss

    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    # Sum number of correct preds

    running_corrects += np.sum(preds.numpy() == labels.numpy())
    # Compute the epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects/ dataset_len
    
    return epoch_loss, epoch_acc



def eval_loop(model, dataloader, device, dataset_len):
    dataset_len = 20
    # Initialize vars
    running_loss = 0.0
    running_corrects = 0


    fin_targets = []
    fin_outputs = []

    new_arr = []

    for i in range(80,100):
      new_arr.append(dataloader.__getitem__(i))

    
    # Loop through eval data
    labels, tweet_features, temporal_features, timestamp = pad_ts_collate(new_arr)
    
    
    output = model.forward(tweet_features, temporal_features, timestamp)

    # Get max of output (not sure why getting indices in torch.max)
    preds = tf.math.argmax(output, 1)

    unique_counts = [len([x for x in np.asarray(labels) if x == 0]), len([x for x in np.asarray(labels) if x == 1])]
    # Calculate loss
    loss = loss_fn(output, labels, unique_counts)
    # Sum loss and correct preds
    running_loss += loss
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    running_corrects += np.sum(preds.numpy() == labels.numpy())

    # Moves the memory back from device to cpu, and converts to numpy
    fin_targets.append(labels.numpy())
    fin_outputs.append(preds.numpy())
    print(preds.numpy())
    # Update loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects / dataset_len

    # Stack each array in fin_outputs and fin_targets in vertical columns
    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets)

# Defines our loss function
def loss_fn(output, targets, samples_per_cls):
    beta = 0.9999
    gamma = 2.0
    no_of_classes = 2
    loss_type = "focal"

    # Class balanced loss 
    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)


def main(config):
    # Defines constants related to our model
    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size

    HIDDEN_DIM = config.hidden_dim
    EMBEDDING_DIM = config.embedding_dim

    NUM_LAYERS = config.num_layer
    DROPOUT = config.dropout
    CURRENT = config.current

    RANDOM = config.random

    DATA_DIR = config.data_dir

    # Different models based on type of config's base_model (defined in model class)
    if config.base_model == "historic":
        model = Historic(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    elif config.base_model == "current":
        model = Current(HIDDEN_DIM, DROPOUT)
    elif config.base_model == "historic-current":
        model = HistoricCurrent(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, config.model)
    else:
        assert False

    # Load train, validation and test data
    with open(os.path.join(DATA_DIR, 'data/samp_data.pkl'), "rb") as f:
        df_train = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'data/samp_data.pkl'), "rb") as f:
        df_val = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'data/samp_data.pkl'), "rb") as f:
        df_test = pickle.load(f)

    # Create datasets (defined in dataloader.py)
    train_dataset = SuicidalDataset(df_train.label.values, df_train.curr_enc.values, df_train.enc.values,
                                    df_train.hist_dates, CURRENT, RANDOM)
    val_dataset = SuicidalDataset(df_val.label.values, df_val.curr_enc.values, df_val.enc.values,
                                  df_val.hist_dates, CURRENT, RANDOM)
    test_dataset = SuicidalDataset(df_test.label.values, df_test.curr_enc.values, df_test.enc.values,
                                   df_test.hist_dates, CURRENT, RANDOM)

    print("TEST: ",df_test.label.values)
    print("train: ",df_train.label.values)
   
    
    # Set device if cpu (ignore cuda) 
    device = 'cpu'
 


    LEARNING_RATE = config.learning_rate


    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)


    # Set name for model based on current time
    model_name = f'{int(datetime.timestamp(datetime.now()))}_{config.base_model}_{config.model}_{config.hidden_dim}_{config.num_layer}_{config.learning_rate}'

   
    # Loop through all epochs
    for epoch in range(EPOCHS):
        # Gets loss and accuracy from training the function, and then from validation data
        loss, accuracy = train_loop(model, train_dataset, optimizer, device, len(train_dataset))
        eval_loss, eval_accuracy, __, _ = eval_loop(model, val_dataset, device, len(val_dataset))


        # Get f1, recall scores
        metric = f1_score(_, __, average="macro")
        recall_1 = recall_score(_, __, average=None)[1]
       

        # Print updates on current epoch
        print(
            f'epoch {epoch + 1}:: train: loss: {loss:.4f}, accuracy: {accuracy:.4f} | valid: loss: {eval_loss:.4f}, accuracy: {eval_accuracy:.4f}, f1: {metric:.4f}, recall_1: {recall_1:.4f}')
        '''if metric > best_metric:
            best_metric = metric
            best_model_wts = copy.deepcopy(model.state_dict())'''

        

   
    if config.test:
        _, _, y_pred, y_true = eval_loop(model, test_dataset, device, len(test_dataset))

        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
        print(report)
        

if __name__ == '__main__':
    # I believe the combinations of the model_set make up the base_model_set
    base_model_set = {"historic", "historic-current", "current"}
    model_set = {"tlstm", "bilstm", "bilstm-attention"}

    # Potential arguments for parser
    parser = argparse.ArgumentParser(description="Temporal Suicidal Modelling")
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-bs", "--batch-size", default=100, type=int)
    parser.add_argument("-e", "--epochs", default=10, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=100, type=int)
    parser.add_argument("-ed", "--embedding-dim", default=768, type=int)
    parser.add_argument("-n", "--num-layer", default=1, type=int)
    parser.add_argument("-d", "--dropout", default=0.5, type=float)
    parser.add_argument("--current", action="store_false")
    parser.add_argument("--base-model", type=str, choices=base_model_set, default="historic-current")
    parser.add_argument("--model", type=str, choices=model_set, default="tlstm")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--random", action="store_true")
    config = parser.parse_args()

    main(config)


