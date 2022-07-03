import math
import time

import tensorflow.keras as keras

from scripts import datasets, models, runtime, toolkit

BS_MULTIPLER = 1

strategy = runtime.prepare_devices(
    mixed_precision=True,
    multi_gpu=False,
    memory_growth=False,
)

ds = datasets.cifar(
    train_batch_size=128 * BS_MULTIPLER,
    valid_batch_size=256 * BS_MULTIPLER,
)


def run(model, LOGDIR):
    with strategy.scope():
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
                keras.metrics.SparseCategoricalCrossentropy(
                    name="ce", from_logits=True
                ),
                toolkit.get_optimizer_learning_rate_metric(optimizer),
            ],
        )

    runtime.save_model_info(model, LOGDIR)

    model.fit(
        ds["train"],
        validation_data=ds["validation"],
        epochs=16,
        steps_per_epoch=math.ceil(4000 / BS_MULTIPLER),
        callbacks=runtime.get_logging_callbacks(LOGDIR),
    )


models_to_test = [
    models.cifar.ResNet20(),
    models.cifar.ResNet56(),
    models.cifar.WideResNet(16, 4),
    models.cifar.WideResNet(16, 8),
]

while models_to_test:
    with strategy.scope():
        LOGDIR = f"output{int(time.time())}"
        model = models_to_test.pop(0)
        run(model, LOGDIR)
