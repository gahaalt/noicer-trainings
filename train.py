import argparse
import importlib
import time
from scripts import runtime

parser = argparse.ArgumentParser()
parser.add_argument(
    "configs",
    nargs="+",
    help="List of all the configs you want to run.",
)
args = parser.parse_args()

for config in args.configs:
    config = config.replace("/", ".")
    if config.endswith(".py"):
        config = config[:-3]
    cfg = importlib.import_module(config)

    initial_epoch = 0
    logdir = f"output{int(time.time())}"

    cfg.model.fit(
        cfg.ds["train"],
        validation_data=cfg.ds["validation"],
        epochs=cfg.num_epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        callbacks=runtime.get_logging_callbacks(logdir),
        initial_epoch=initial_epoch,
    )
