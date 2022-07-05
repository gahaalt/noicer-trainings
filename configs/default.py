from scripts import runtime
batch_multipler = 1

strategy = runtime.prepare_devices(
    mixed_precision=True,
    memory_growth=False,
    multi_gpu=False,
)
