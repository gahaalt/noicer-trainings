from scripts import datasets

from ..default import *

ds = datasets.mnist(
    train_batch_size=100 * batch_multipler,
    valid_batch_size=200 * batch_multipler,
)