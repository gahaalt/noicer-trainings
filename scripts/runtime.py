import os
import shutil

from collections import Counter
import tensorflow as tf
from tensorflow import keras
from termcolor import colored, cprint

from scripts import toolkit

LOGGING_KWDS = {"color": "green", "attrs": ["bold", "dark"]}


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
    cprint(f"MODEL INFO", **LOGGING_KWDS)
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
    cprint(f"LAYER COUNTS: {dict(layer_counts)}", **LOGGING_KWDS)

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

    cprint(f"TRAINABLE WEIGHTS: {trainable_w}", **LOGGING_KWDS)
    cprint(
        f"KERNELS: {kernels} ({kernels / trainable_w * 100:^6.2f}%), "
        f"BIASES: {biases} ({biases / trainable_w * 100:^6.2f}%), "
        f"BN: {bn} ({bn / trainable_w * 100:^6.2f}%)",
        **LOGGING_KWDS,
    )


def save_model_info(model, directory):
    print_model_info(model)

    # we will have a nice diagram of model in the output directory
    os.makedirs(directory, exist_ok=True)

    model_diagram_path = os.path.join(directory, "model.png")
    cprint(f"SAVING MODEL DIAGRAM TO `{model_diagram_path}`", **LOGGING_KWDS)
    keras.utils.plot_model(
        model,
        to_file=model_diagram_path,
        show_shapes=True,
        show_layer_names=False,
    )

    # we copy original source code to have reproducibile output!
    # this way we can 100% restore the model that was trained
    codebase_path = os.path.join(directory, "scripts")
    cprint(f"SAVING CODEBASE TO `{codebase_path}`", **LOGGING_KWDS)
    shutil.copytree(
        "scripts",
        codebase_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )


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
