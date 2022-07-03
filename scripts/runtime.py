import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from termcolor import cprint

from scripts import toolkit

INFO_KWDS = {"color": "green", "attrs": ["bold", "dark"]}


def prepare_devices(
    mixed_precision=True,
    multi_gpu=False,
    memory_growth=True,
):
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    if memory_growth:
        devices = tf.config.list_physical_devices("GPU")
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)

    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    return strategy


def print_model_info(model):
    layer_counts = Counter()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_counts["Dense"] += 1
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_counts["Conv2D"] += 1
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer_counts["BatchNorm"] += 1
        if isinstance(layer, tf.keras.layers.Dropout):
            layer_counts["Dropout"] += 1
    cprint(f"LAYERS IN THE MODEL: {dict(layer_counts)}", **INFO_KWDS)

    bn = 0
    biases = 0
    kernels = 0
    trainable_w = 0
    for w in model.trainable_weights:
        n = w.shape.num_elements()
        trainable_w += n

    for layer in model.layers:
        if hasattr(layer, "beta") and layer.beta is not None:
            bn += layer.beta.shape.num_elements()

        if hasattr(layer, "gamma") and layer.gamma is not None:
            bn += layer.gamma.shape.num_elements()

        if hasattr(layer, "bias") and layer.bias is not None:
            biases += layer.bias.shape.num_elements()

        if hasattr(layer, "kernel"):
            kernels += layer.kernel.shape.num_elements()

    cprint(f"TRAINABLE WEIGHTS: {trainable_w}", **INFO_KWDS)
    cprint(
        f"KERNELS: {kernels} ({kernels / trainable_w * 100:^6.2f}%), "
        f"BIASES: {biases} ({biases / trainable_w * 100:^6.2f}%), "
        f"BN: {bn} ({bn / trainable_w * 100:^6.2f}%)",
        **INFO_KWDS,
    )


def save_model_info(model, directory):
    print_model_info(model)

    # we will have a nice diagram of model in the output directory
    os.makedirs(directory, exist_ok=True)

    model_diagram_path = os.path.join(directory, "model.png")
    cprint(f"SAVING MODEL DIAGRAM TO `{model_diagram_path}`", **INFO_KWDS)
    keras.utils.plot_model(
        model,
        to_file=model_diagram_path,
        show_shapes=True,
        show_layer_names=False,
    )

    # we copy original source code to have reproducibile output!
    # this way we can 100% restore the model that was trained
    codebase_path = os.path.join(directory, "scripts")
    cprint(f"SAVING CODEBASE TO `{codebase_path}`", **INFO_KWDS)
    shutil.copytree(
        "scripts",
        codebase_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    try:
        script_name = Path(sys.argv[0]).name
        shutil.copy(script_name, directory)
        cprint(
            f"RESUME BY RUNNING `{os.path.join(directory, script_name)}`", **INFO_KWDS
        )
    except NameError:
        pass


def get_logging_callbacks(directory, profiling=False):
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=directory,
        histogram_freq=1,
        profile_batch=(3, 5) if profiling else 0,
    )
    checkpoint_callback = toolkit.CheckpointAfterEpoch(
        directory_path=directory,
        keep_only_latest_checkpoint=False,
    )
    return tensorboard_callback, checkpoint_callback


def load_checkpoint_if_available(model, optimizer):
    files = " ".join(os.listdir("."))
    matches = re.findall(r"checkpoint_ep(\d+)", files)
    if not matches:
        cprint(f"CHECKPOINT NOT FOUND! TRAINING FROM SCRATCH...", **INFO_KWDS)
        return None

    latest_epoch = max([int(num) for num in matches])

    checkpoint_h5 = f"checkpoint_ep{latest_epoch}.h5"
    checkpoint_pkl = f"checkpoint_ep{latest_epoch}.pkl"
    assert checkpoint_h5 in os.listdir("."), "MISSING h5 CHECKPOINT"
    assert checkpoint_pkl in os.listdir("."), "MISSING pkl CHECKPOINT"

    cprint(f"LOADING CHECKPOINTS: {checkpoint_h5} {checkpoint_pkl}", **INFO_KWDS)
    toolkit.reset_weights_to_checkpoint(model, checkpoint_h5)

    # need to build optimizer otherwise optimizer.weights = [] and can't load
    toolkit.build_optimizer(model, optimizer)
    toolkit.load_optimizer(optimizer, checkpoint_pkl)
    return latest_epoch
