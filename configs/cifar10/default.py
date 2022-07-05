from scripts import datasets

from ..default import *

ds = datasets.cifar(
    train_batch_size=128 * batch_multipler,
    valid_batch_size=256 * batch_multipler,
)