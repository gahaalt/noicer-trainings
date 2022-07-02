import os
import pickle
from collections import Counter, abc

import numpy as np
import tensorflow as tf


def get_kernels(model):
    return [l.kernel for l in model.layers if hasattr(l, 'kernel')]


def clip_many(values, clip_at, clip_from=None, inplace=False):
    """Clips a list of tf or np arrays. Returns tf arrays."""

    if clip_from is None:
        clip_from = -clip_at

    if inplace:
        for v in values:
            v.assign(tf.clip_by_value(v, clip_from, clip_at))
    else:
        r = []
        for v in values:
            r.append(tf.clip_by_value(v, clip_from, clip_at))
        return r


def concatenate_flattened(arrays):
    return np.concatenate([x.flatten() if isinstance(x, np.ndarray)
                           else x.numpy().flatten() for x in arrays], axis=0)


def set_all_weights_from_model(model, source_model):
    """Warning if a pair doesn't match."""

    for w1, w2 in zip(model.weights, source_model.weights):
        if w1.shape == w2.shape:
            w1.assign(w2)
        else:
            print(f"WARNING: Skipping {w1.name}: {w1.shape} != {w2.shape}")


def clone_model(model):
    """tf.keras.models.clone_model + toolkit.set_all_weights_from_model"""

    new_model = tf.keras.models.clone_model(model)
    set_all_weights_from_model(new_model, model)
    return new_model


def reset_weights_to_checkpoint(model, ckp=None, skip_keyword=None):
    """Reset network in place, has an ability to skip keybword."""

    temp = tf.keras.models.clone_model(model)
    if ckp:
        temp.load_weights(ckp)
    skipped = 0
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword and skip_keyword in w1.name:
            skipped += 1
            continue
        w1.assign(w2)
    print(f"INFO RESET: Skipped {skipped} layers with keyword {skip_keyword}!")
    return skipped


def get_optimizer_learning_rate_metric(opt):
    if hasattr(opt, '_decayed_lr'):
        def lr(*args):
            return opt._decayed_lr(tf.float32)

        return lr
    else:
        raise UserWarning("MISSING _decayed_lr!")


def save_optimizer(optimizer, path):
    if not path.endswith(".pkl"):
        path = path + ".pkl"

    if dirpath := os.path.dirname(path):
        os.makedirs(dirpath, exist_ok=True)
    weights = optimizer.get_weights()
    with open(path, 'wb') as f:
        pickle.dump(weights, f)


def build_optimizer(model, optimizer):
    zero_grad = [tf.zeros_like(w) for w in model.trainable_weights]
    optimizer.apply_gradients(zip(zero_grad, model.trainable_weights))


def load_optimizer(optimizer, path):
    with open(path, 'rb') as f:
        weights = pickle.load(f)
    try:
        optimizer.set_weights(weights)
    except ValueError as e:
        print("!!!WARNING!!! Tried to load empty optimizer!")
        print(e)


def save_model(model, path):
    if not path.endswith(".h5"):
        path = path + ".h5"
    if dirpath := os.path.dirname(path):
        os.makedirs(dirpath, exist_ok=True)
    model.save_weights(path, save_format="h5")


class CheckpointAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, directory_path, keep_only_latest_checkpoint=True):
        super().__init__()
        self.directory_path = directory_path
        self.keep_only_latest = keep_only_latest_checkpoint

    def on_epoch_end(self, epoch, logs=None):
        path = os.path.join(self.directory_path, f"checkpoint_ep{epoch}")
        save_model(self.model, path)
        save_optimizer(self.model.optimizer, path)
        if self.keep_only_latest and epoch > 0:
            prev_path = os.path.join(self.directory_path, f"checkpoint_ep{epoch - 1}")
            os.remove(prev_path + ".h5")
            os.remove(prev_path + ".pkl")
