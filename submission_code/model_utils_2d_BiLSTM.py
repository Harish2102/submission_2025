import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, LayerNormalization, Activation, Add,
    Concatenate, Dropout, MultiHeadAttention, LSTM, Bidirectional,
    Reshape, Permute, Dense
)

# Scikit-learn
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Signal processing
from biosppy.signals import ecg

# Project-specific utilities
from _preprocess_utils_2d import *



# Helper functions


def transformer_block(input_tensor, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    # Self-Attention Layer
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim)(input_tensor, input_tensor)
    attention_output = Dropout(dropout_rate)(attention_output)

    attention_residual = Add()([input_tensor, attention_output])
    attention_output = LayerNormalization()(attention_residual)

    # Feedforward Network
    ff_output = Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = Dense(input_tensor.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)

    # Residual Connection and Layer Normalization
    ff_residual = Add()([attention_output, ff_output])
    transformer_output = LayerNormalization()(ff_residual)

    return transformer_output

# Multi-scale convolutional blocks for feature extraction


def multi_scale_conv(input_features, num_filters, filter_sizes=[3, 5, 7], layer_name="multi_scale"):
    conv_3 = Conv1D(num_filters, filter_sizes[0], padding='same', activation="gelu",
                    dtype='float32', name=f'{layer_name}_conv3')(input_features)
    conv_5 = Conv1D(num_filters, filter_sizes[1], padding='same', activation="gelu",
                    dtype='float32', name=f'{layer_name}_conv5')(input_features)
    conv_7 = Conv1D(num_filters, filter_sizes[2], padding='same', activation="gelu",
                    dtype='float32', name=f'{layer_name}_conv7')(input_features)

    c1 = Concatenate(dtype='float32', name=f'{layer_name}_concat1')(
        [conv_3, conv_5])
    c2 = Concatenate(dtype='float32', name=f'{layer_name}_concat2')(
        [conv_5, conv_7])
    c3 = Concatenate(dtype='float32', name=f'{layer_name}_concat3')(
        [conv_3, conv_7])

    c11 = Conv1D(num_filters, 1, padding='same', activation="gelu",
                 dtype='float32', name=f'{layer_name}_conv35')(c1)
    c12 = Conv1D(num_filters, 1, padding='same', activation="gelu",
                 dtype='float32', name=f'{layer_name}_conv57')(c2)
    c13 = Conv1D(num_filters, 1, padding='same', activation="gelu",
                 dtype='float32', name=f'{layer_name}_conv37')(c3)

    concat = Concatenate(
        dtype='float32', name=f'{layer_name}_concat')([c11, c12, c13])
    fin_conv = Conv1D(num_filters, 1, padding='same', activation="gelu",
                      dtype='float32', name=f'{layer_name}_finconv')(concat)

    return fin_conv


def ecg_model(seq_len: int = 4096, n_leads: int = 12,
              num_filters=(6, 3, 1),
              lstm_units: int = 64,
              bidirectional: bool = True):

    inp = Input(shape=(seq_len, n_leads), name="ecg12")

    # ── CONV BLOCK 1 ───────────────────────────────────────────
    x = multi_scale_conv(inp, num_filters[0], layer_name="enc_b11")
    x = multi_scale_conv(x,   num_filters[0], layer_name="enc_b12")
    x = MaxPool1D(2)(x)                      # (seq_len/2, filters)

    # ── CONV BLOCK 2 ───────────────────────────────────────────
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = multi_scale_conv(x, num_filters[1], layer_name="enc_b21")
    x = multi_scale_conv(x, num_filters[1], layer_name="enc_b22")
    x = MaxPool1D(2)(x)                      # (seq_len/4, filters)

    # ── CONV BLOCK 3 ───────────────────────────────────────────
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = multi_scale_conv(x, num_filters[2], layer_name="enc_b31")
    x = multi_scale_conv(x, num_filters[2], layer_name="enc_b32")
    x = MaxPool1D(2)(x)                      # (seq_len/8, filters)

    # Now x has shape (batch, T≈512, C)
    # ── (Optional) Multi-Head Self-Attention on the sequence ──
    x = MultiHeadAttention(num_heads=4, key_dim=16,
                           name="mhsa")(x, x)

    # ── LSTM / Bi-LSTM ─────────────────────────────────────────
    if bidirectional:
        x = Bidirectional(
                LSTM(lstm_units,
                     return_sequences=False,
                     dropout=0.2,
                     recurrent_dropout=0.1),
                name="bilstm")(x)
    else:
        x = LSTM(lstm_units,
                 return_sequences=False,
                 dropout=0.2,
                 recurrent_dropout=0.1,
                 name="lstm")(x)

    # ── Dense head ─────────────────────────────────────────────
    x = Dense(128, activation='gelu')(x)
    x = Dense(64,  activation='gelu')(x)
    x = Dense(16,  activation='gelu')(x)
    x = Dropout(0.3)(x)
    x = Dense(8,   activation='gelu')(x)

    out = Dense(1, activation='sigmoid', name="out_prob")(x)

    return Model(inp, out, name="ECG_ConvAttn_LSTM")

def focal_loss(gamma: float = 2.0):
    """Focal Loss (Lin et al.) for binary problems with sigmoid output."""
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = K.binary_crossentropy(y_true, y_pred, from_logits=False)          # already mean-reduced per-sample
        p_t = tf.clip_by_value(y_true * y_pred + (1-y_true)*(1-y_pred), 1e-7, 1-1e-7)
        focal = K.pow(1.0 - p_t, gamma) * bce
        return focal
    return _loss

class DynamicThreshold(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, threshold_var):
        super().__init__()
        self.val_ds = val_ds
        self.threshold_var = threshold_var

    @tf.function  # speeds up large val sets
    def _collect(self, x):
        return self.model(x, training=False)

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_score = [], []
        for x, y in self.val_ds:
            y_score.append(self._collect(x))
            y_true.append(tf.cast(y, tf.float32))

        y_true  = tf.concat(y_true,  axis=0).numpy().ravel()
        y_score = tf.concat(y_score, axis=0).numpy().ravel()

        fpr, tpr, thresh = roc_curve(y_true, y_score)
        finite = np.isfinite(thresh)         # skip the +inf sentinel
        best = np.argmax(tpr[finite] - fpr[finite])

        best_thr = float(thresh[finite][best])
        self.threshold_var.assign(best_thr)

        if logs is not None:
            logs["val_best_threshold"] = best_thr


class TopKTPR(tf.keras.metrics.Metric):
    def __init__(self, k_fraction=0.05, name="tpr_at_top5", **kw):
        super().__init__(name=name, **kw)
        self.k_fraction = float(k_fraction)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.p  = self.add_weight(name="p",  initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten (works for any output shape)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        n = tf.cast(tf.size(y_pred), tf.float32)
        k = tf.cast(tf.math.maximum(1.0, tf.math.ceil(n * self.k_fraction)), tf.int32)

        # ---- efficient mask: no full scatter for large n ----
        topk_idx = tf.math.top_k(y_pred, k=k, sorted=False).indices
        topk_true = tf.gather(y_true, topk_idx)

        self.tp.assign_add(tf.reduce_sum(topk_true))
        self.p.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return self.tp / (self.p + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.p.assign(0.0)


def f1_metric(threshold_var):
    def _f1(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(tf.greater_equal(y_pred, threshold_var), tf.float32)

        tp = tf.reduce_sum(y_true * y_pred_bin)
        fn = tf.reduce_sum(y_true * (1 - y_pred_bin))
        fp = tf.reduce_sum((1 - y_true) * y_pred_bin)

        precision = tp / (tp + fp + K.epsilon())
        recall    = tp / (tp + fn + K.epsilon())
        return 2 * precision * recall / (precision + recall + K.epsilon())
    return _f1