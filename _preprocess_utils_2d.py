import wfdb
import numpy as np
import tensorflow as tf
import glob
from scipy.signal import butter, filtfilt, iirnotch
from biosppy.signals import ecg
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


TARGET_FS   = 400        # final sampling rate (Hz)
TARGET_LEN  = 4096       # final signal length (samples)
HP_CUTOFF   = 0.5        # high-pass corner (Hz)
NOTCH_FREQ  = 50         # power-line frequency (Hz)
NOTCH_Q     = 30         # notch filter Q
HP_ORDER    = 4

nyq            = 0.5 * TARGET_FS
hp_norm        = HP_CUTOFF / nyq
hp_b, hp_a     = butter(HP_ORDER, hp_norm, btype="high", analog=False)
notch_b, notch_a = iirnotch(NOTCH_FREQ / nyq, NOTCH_Q)

def data_preprocess(signal, fs):
    """
    signal : np.ndarray (n_samples, n_leads)  – WFDB p_signal
    fs     : int or float                     – original sampling rate (Hz)
    returns: np.ndarray (TARGET_LEN, 12)      – float32, zero-padded/truncated
    """
    # ------------------------------------------------------------------ #
    # 0) Resample if needed
    # ------------------------------------------------------------------ #
    if fs != TARGET_FS:
        gcd  = np.gcd(int(fs), TARGET_FS)         # integer resample ratio
        up   = TARGET_FS // gcd
        down = int(fs) // gcd
        # resample every lead in one shot (axis 0 = time)
        signal = resample_poly(signal, up, down, axis=0).astype(np.float32)
        fs = TARGET_FS

    signal = signal.T                              # → shape (n_leads, n_samples)

    # ------------------------------------------------------------------ #
    # 1) Filters (per lead)
    # ------------------------------------------------------------------ #


    filtered = filtfilt(hp_b, hp_a, signal, axis=1)
    filtered = filtfilt(notch_b, notch_a, filtered, axis=1)

    # ------------------------------------------------------------------ #
    # 2) Normalize (z-score per lead)
    # ------------------------------------------------------------------ #
    eps   = 1e-12
    mean  = filtered.mean(axis=1, keepdims=True)
    std   = filtered.std(axis=1,  keepdims=True) + eps
    norm  = (filtered - mean) / std

    # ------------------------------------------------------------------ #
    # 3) Pad / truncate to TARGET_LEN
    # ------------------------------------------------------------------ #
    cur_len = norm.shape[1]
    if cur_len < TARGET_LEN:                       # pad right with zeros
        pad = np.zeros((norm.shape[0], TARGET_LEN - cur_len), dtype=norm.dtype)
        norm = np.concatenate((norm, pad), axis=1)
    elif cur_len > TARGET_LEN:
        norm = norm[:, :TARGET_LEN]

    return norm.T.astype(np.float32)               # shape (TARGET_LEN, 12)


def tf_dataloader(record_paths, batch_size=64, shuffle=True):
    def _preprocess_and_load(file_path):
        def _py_load(path_bytes):

            path = path_bytes.numpy().decode('utf-8')

            record = wfdb.rdrecord(path, physical=True)
            fs = record.fs

            chagas_label = False
            for c in record.comments:
                if c.startswith('Chagas label:'):
                    parts = c.split(':')
                    if len(parts) == 2:
                        label_str = parts[1].strip().lower()
                        chagas_label = (label_str == 'true')
                    break
            label = np.int32(chagas_label)

            filtered_signal = data_preprocess(
                record.p_signal.astype(np.float32), fs)
            return filtered_signal, label

        processed_signal, label = tf.py_function(
            func=_py_load,
            inp=[file_path],
            Tout=(tf.float32, tf.int32)
        )

        processed_signal.set_shape((4096, 12))
        label.set_shape(())
        return processed_signal, label

    dataset = tf.data.Dataset.from_tensor_slices(record_paths)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(
            record_paths), reshuffle_each_iteration=True)
    dataset = dataset.map(_preprocess_and_load,
                          num_parallel_calls=tf.data.AUTOTUNE)

    # dataset = dataset.ignore_errors()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
