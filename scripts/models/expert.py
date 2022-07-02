"""
Creating ResNetV2.
"""

import tensorflow.keras as keras

LAYER_COUNTER = 0
GLOBAL_REGULARIZER = 0


def simple_block(block_input, output_filters, stride):
    flow = conv2d(
        output_filters,
        3,
        strides=stride,
    )(block_input)

    flow = preactivation(flow)
    flow = conv2d(output_filters, 3, use_bias=True)(flow)
    return flow


def bottleneck_block(block_input, output_filters, stride):
    flow = conv2d(output_filters, 1)(block_input)

    flow = preactivation(flow)
    flow = conv2d(
        output_filters,
        3,
        strides=stride,
    )(flow)

    flow = preactivation(flow)
    flow = conv2d(output_filters * 4, 1, use_bias=True)(flow)
    return flow


def shortcut(residual, previous_value, preactivated_value, stride):
    global LAYER_COUNTER
    if residual.shape[-1] != previous_value.shape[-1]:
        value = preactivated_value
    else:
        value = previous_value

    if stride != 1:
        value = keras.layers.MaxPool2D(pool_size=stride, padding="same")(value)

    if residual.shape[-1] != previous_value.shape[-1]:
        value = conv2d(
            filters=residual.shape[-1],
            kernel_size=1,
            name=f"SHORTCUT{LAYER_COUNTER}",
            use_bias=True,
        )(value)
        LAYER_COUNTER += 1
    return keras.layers.Add()([residual, value])


def conv2d(*args, **kwds):
    kwds.setdefault("use_bias", False)
    return keras.layers.Conv2D(
        *args,
        **kwds,
        padding="same",
        kernel_regularizer=GLOBAL_REGULARIZER,
    )


def preactivation(flow):
    flow = keras.layers.BatchNormalization()(flow)
    flow = keras.layers.ReLU()(flow)
    return flow


version_to_size = {
    # IMAGENET
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    # CIFAR
    20: (3, 3, 3),
    32: (5, 5, 5),
    44: (7, 7, 7),
    56: (9, 9, 9),
    110: (18, 18, 18),
}


def resnet2(
    input_shape,
    n_classes,
    block_type,  # "simple" or "bottleneck"
    towers_filters,
    towers_num_blocks,
    l2_regularization,
    num_downsamplings,  # 2 for imagenet, 0 for small networks
    first_conv_kernel_size,
    towers_strides=None,  # stride is applied at the END of the tower
    what_to_output=("logits",),  # logits, feature_map, feature_map1...
):
    ############ PARSING

    if towers_filters == "IMAGENET":
        towers_filters = (64, 128, 256, 512)
        num_downsamplings = 2
    elif towers_filters == "CIFAR":
        towers_filters = (16, 32, 64)
        num_downsamplings = 0

    num_towers = len(towers_filters)

    if isinstance(towers_num_blocks, int):
        towers_num_blocks = (towers_num_blocks,) * num_towers

    if towers_strides is None:
        towers_strides = (2,) * (num_towers - 1) + (1,)

    #############

    global GLOBAL_REGULARIZER
    if l2_regularization:
        GLOBAL_REGULARIZER = keras.regularizers.l2(l2_regularization)

    model_input = keras.layers.Input(shape=input_shape)
    flow = conv2d(
        filters=towers_filters[0],
        kernel_size=first_conv_kernel_size,
        strides=2 if num_downsamplings >= 1 else 1,
        use_bias=num_downsamplings > 1,  # we want to use it if maxpooling is next
    )(model_input)
    if num_downsamplings > 1:
        stride = 2 ** (num_downsamplings - 1)
        flow = keras.layers.MaxPool2D(
            pool_size=stride + 1,
            strides=stride,
            padding="same",
        )(flow)

    if block_type.startswith("bo"):
        block_fn = bottleneck_block
    else:
        block_fn = simple_block

    feature_maps = []

    for tower_size, num_filters, stride in zip(
            towers_num_blocks, towers_filters, towers_strides
    ):
        for idx_in_tower in range(tower_size):
            stage_input = flow
            preactivated_value = preactivation(stage_input)

            stride_now = stride if idx_in_tower == tower_size - 1 else 1

            flow = block_fn(
                block_input=preactivated_value,
                output_filters=num_filters,
                stride=stride_now,
            )
            flow = shortcut(flow, stage_input, preactivated_value, stride_now)
        feature_maps.append(flow)

    flow = keras.layers.BatchNormalization()(flow)
    flow = keras.layers.ReLU()(flow)

    flow = keras.layers.GlobalAvgPool2D()(flow)
    output = keras.layers.Dense(
        units=n_classes,
        kernel_regularizer=GLOBAL_REGULARIZER,
    )(flow)

    all_output_options = {
        "logits": output,
    }
    for idx, fmap in enumerate(feature_maps):
        all_output_options[f"feature_map{idx}"] = fmap
        all_output_options["feature_map"] = fmap

    model = keras.models.Model(
        inputs=model_input,
        outputs=[all_output_options[o] for o in what_to_output],
    )
    return model
