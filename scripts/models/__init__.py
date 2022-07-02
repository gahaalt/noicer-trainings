from scripts.models.expert import resnet2


class Cifar:
    def __init__(self):
        self.kwds = dict(
            input_shape=(32, 32, 3),
            n_classes=10,
            block_type="simple",
            towers_filters=(16, 32, 64),
            l2_regularization=1e-4,
            num_downsamplings=0,
            first_conv_kernel_size=3,
            towers_strides=None,
            what_to_output=("logits",),
        )

    def ResNet20(self, **kwds):
        new_kwds = dict(towers_num_blocks=3)
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet32(self, **kwds):
        new_kwds = dict(towers_num_blocks=5)
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet44(self, **kwds):
        new_kwds = dict(towers_num_blocks=7)
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet56(self, **kwds):
        new_kwds = dict(towers_num_blocks=9)
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet110(self, **kwds):
        new_kwds = dict(towers_num_blocks=18)
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def WideResNet16_4(self, **kwds):
        new_kwds = dict(
            towers_filters=(16 * 4, 32 * 4, 64 * 4),
            towers_num_blocks=2,
            l2_regularization=2e-4,
        )
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def WideResNet(self, N, K, **kwds):
        assert (N - 4) % 6 == 0
        tower_size = int((N - 4) / 6)
        new_kwds = dict(
            towers_filters=(16 * K, 32 * K, 64 * K),
            towers_num_blocks=tower_size,
            l2_regularization=2e-4,
        )
        return resnet2(**{**self.kwds, **new_kwds, **kwds})


class Cifar100(Cifar):
    def __init__(self):
        super().__init__()
        self.kwds["n_classes"] = 100


class Mnist(Cifar):
    def __init__(self):
        super().__init__()
        self.kwds["input_shape"] = (28, 28, 1)


class Imagenet:
    def __init__(self):
        self.kwds = dict(
            input_shape=(224, 224, 3),
            n_classes=1000,
            block_type="bottleneck",
            towers_filters=(64, 128, 256, 512),
            l2_regularization=1e-4,
            num_downsamplings=2,
            first_conv_kernel_size=7,
            towers_strides=None,
            what_to_output=("logits",),
        )

    def ResNet18(self, **kwds):
        new_kwds = dict(towers_num_blocks=(2, 2, 2, 2), block_type="simple")
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet34(self, **kwds):
        new_kwds = dict(towers_num_blocks=(3, 4, 6, 3), block_type="simple")
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet50(self, **kwds):
        new_kwds = dict(towers_num_blocks=(3, 4, 6, 3))
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet101(self, **kwds):
        new_kwds = dict(towers_num_blocks=(3, 4, 23, 3))
        return resnet2(**{**self.kwds, **new_kwds, **kwds})

    def ResNet152(self, **kwds):
        new_kwds = dict(towers_num_blocks=(3, 8, 36, 3))
        return resnet2(**{**self.kwds, **new_kwds, **kwds})


cifar = Cifar()
cifar100 = Cifar100()
mnist = Mnist()
imagenet = Imagenet()
