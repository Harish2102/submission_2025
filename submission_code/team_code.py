#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *
from model_utils_2d_BiLSTM import *
from _preprocess_utils_2d import *
import tensorflow as tf
import os

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    return None

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_file, verbose):
    model = ecg_model()
    threshold_var = tf.Variable(0.5, dtype=tf.float32, trainable=False, name="dyn_thresh")

    model.compile(
        optimizer='adam',
        loss=focal_loss(gamma=2.0),  # Use focal loss
        metrics=['accuracy',
                tf.keras.metrics.AUC(name="auc"),
                f1_metric(threshold_var),
                TopKTPR(k_fraction=0.05)]
    )

    weight_pt = model_file
    model.load_weights(weight_pt)
    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
THRESHOLD = 0.2143  # dynamic threshold from validation tuning

def run_model(record, model, verbose):

    record_data = wfdb.rdrecord(record, physical=True)
    fs = record_data.fs
    signal = record_data.p_signal.astype(np.float32)
    x = data_preprocess(signal, fs)
    x = np.expand_dims(x, axis=0)

    prob = float(model.predict(x, verbose=0)[0])
    binary = int(prob > THRESHOLD)

    if verbose:
        print(f"Binary Output: {binary}, Probability Output: {prob} (Threshold = {THRESHOLD})")

    return binary, prob