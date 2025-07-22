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
import sys

from helper_code import *
from model_utils_2d_BiLSTM import *
from _preprocess_utils_2d import *
import tensorflow as tf
import os
from pathlib import Path

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
MODEL_FILENAME = "trained_model.keras"

def load_model(model_folder, verbose=False):
    # 1) Candidates to look in
    candidates = [
        Path(model_folder),                               # what run_model.py passed
        Path(__file__).resolve().parent / "model",        # repo_root/model
        Path("/model"),                                   # PhysioNet default mount
    ]

    # 2) Pick the first directory that actually has the file
    model_path = None
    for d in candidates:
        p = d / MODEL_FILENAME
        if p.exists():
            model_path = p
            break

    if model_path is None:
        if verbose:
            print("Checked dirs:", [str(d) for d in candidates])
        raise FileNotFoundError(f"{MODEL_FILENAME} not found in any candidate directory.")

    if verbose:
        print("Loading model from:", model_path)

    # 3) Try loading as a full Keras model first
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception:
        # Fallback: you only saved weights; rebuild the architecture and load them
        model = ecg_model()
        threshold_var = tf.Variable(0.5, dtype=tf.float32, trainable=False, name="dyn_thresh")
        model.compile(
            optimizer="adam",
            loss=focal_loss(gamma=2.0),
            metrics=["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     f1_metric(threshold_var),
                     TopKTPR(k_fraction=0.05)]
        )
        model.load_weights(str(model_path))
        return model

def run_model(record, model, verbose):
    record_data = wfdb.rdrecord(record, physical=True)
    fs = record_data.fs
    signal = record_data.p_signal.astype(np.float32)
    x = data_preprocess(signal, fs)
    x = np.expand_dims(x, axis=0)
    THRESHOLD = 0.2143  # This is the threshold used in the original model

    prob = float(model.predict(x, verbose=0)[0])
    binary = int(prob > THRESHOLD)
    
    if verbose:
        print(f"Binary Output: {binary}, Probability Output: {prob} (Threshold = {THRESHOLD})")

    return binary, prob