import tensorflow.keras as keras

from scripts import models

from .default import *

with strategy.scope():
    model = models.cifar.ResNet32()

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )
