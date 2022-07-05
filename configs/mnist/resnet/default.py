import tensorflow.keras as keras

from scripts import toolkit

from ..default import *

num_epochs = 16
batch_multipler = 1
steps_per_epoch = 4000

with strategy.scope():
    schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=(
            400 / batch_multipler,
            32000 / batch_multipler,
            48000 / batch_multipler,
        ),
        values=(
            0.001 * batch_multipler,  # WARMUP
            0.1 * batch_multipler,
            0.01 * batch_multipler,
            0.001 * batch_multipler,
        ),
    )

    optimizer = keras.optimizers.SGD(
        learning_rate=schedule,
        momentum=0.9,
        nesterov=True,
    )

metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top3acc"),
    keras.metrics.SparseCategoricalCrossentropy(name="ce", from_logits=True),
    toolkit.get_optimizer_learning_rate_metric(optimizer),
]
