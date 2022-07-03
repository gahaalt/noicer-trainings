import math
import time

import tensorflow.keras as keras

from scripts import datasets, models, runtime, toolkit

BS_MULTIPLER = 1

strategy = runtime.prepare_devices(
    mixed_precision=True,
    memory_growth=False,
    multi_gpu=False,
)

ds = datasets.cifar(
    train_batch_size=128 * BS_MULTIPLER,
    valid_batch_size=256 * BS_MULTIPLER,
)

with strategy.scope():
    model = models.cifar.ResNet20()

    schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=(
            400 / BS_MULTIPLER,
            32000 / BS_MULTIPLER,
            48000 / BS_MULTIPLER,
        ),
        values=(
            0.001 * BS_MULTIPLER,  # WARMUP
            0.1 * BS_MULTIPLER,
            0.01 * BS_MULTIPLER,
            0.001 * BS_MULTIPLER,
        ),
    )
    optimizer = keras.optimizers.SGD(
        learning_rate=schedule,
        momentum=0.9,
        nesterov=True,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top3acc"),
            keras.metrics.SparseCategoricalCrossentropy(name="ce", from_logits=True),
            toolkit.get_optimizer_learning_rate_metric(optimizer),
        ],
    )

initial_epoch = runtime.load_checkpoint_if_available(model, optimizer)
if initial_epoch is None:
    logdir = f"output{int(time.time())}"
    runtime.save_model_info(model, logdir)
    initial_epoch = 0
else:
    logdir = "."

model.fit(
    ds["train"],
    validation_data=ds["validation"],
    epochs=16,
    steps_per_epoch=math.ceil(4000 / BS_MULTIPLER),
    callbacks=runtime.get_logging_callbacks(logdir),
    initial_epoch=initial_epoch,
)
