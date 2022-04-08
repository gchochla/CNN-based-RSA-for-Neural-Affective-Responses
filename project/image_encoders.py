from typing import Optional, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    googlenet,
    inception_v3,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg16_bn,
    vgg11_bn,
    vgg13_bn,
    vgg19_bn,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
)


class Embeddings(nn.Module):
    """Base image encoder class.

    Attributes:
        id: if the current encoder comes in different flavors,
            this distinguishes one version from the other, e.g.
            ResNet-152 vs ResNet-101. Implemented as function
            with property decorator.
        out_dim: dimensionality of output. Implemented as function
            with property decorator.
    """

    def __init__(self):
        super().__init__()

    @property
    def id(self):
        raise NotImplementedError

    @property
    def out_dim(self):
        raise NotImplementedError


class ResNetEmbeddings(Embeddings):
    """ResNet feature extractor.

    Features are extracted by Average Pooling layer of ResNets.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        See `Embeddings`.
        features: net doing the feature extraction.
        resnet_ids_512: ResNet IDs that result in 512-dimensional
            representations.
        resnet_ids_2048: Same for 2048.
        resnet_out_dim: mapping from resnet IDs (int or str) to
            output dim.
    """

    resnet_ids_512 = [18, 34]
    resnet_ids_2048 = [50, 101, 152]
    resnet_out_dim = {id: 512 for id in resnet_ids_512}
    resnet_out_dim.update({str(id): 512 for id in resnet_ids_512})
    resnet_out_dim.update({id: 2048 for id in resnet_ids_2048})
    resnet_out_dim.update({str(id): 2048 for id in resnet_ids_2048})

    def __init__(self, resnet_id: Optional[Union[int, str]] = 152):
        """Init.
        Args:
            resnet_id: ResNet version to load,
                default is 152 (i.e. ResNet152).
        """

        super().__init__()

        resnet_ids = self.resnet_ids_512 + self.resnet_ids_2048
        assert resnet_id in resnet_ids + list(map(str, resnet_ids))

        self.resnet_id = resnet_id

        resnet = globals()["resnet{}".format(resnet_id)](
            pretrained=True, progress=False
        )
        # thankfully resnet's forward is just a series of forward propagations
        # https://github.com/pytorch/vision/blob/39772ece7c87ab38b9c2b9df8e7c85e967a739de/torchvision/models/resnet.py#L264
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    @property
    def id(self):
        return self.resnet_id

    @property
    def out_dim(self):
        return self.resnet_out_dim[self.resnet_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input images.

        Returns:
            Embeddings.
        """

        x = self.features(x)
        x = x.flatten(1)
        return x


class Inception3Embeddings(Embeddings):
    """Inception v3 feature extractor.

    Features are extracted from Average pooling layer of Inception v3.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        See `Embeddings`.
        features: the whole Inception model.
    """

    def __init__(self):
        """Init."""
        super().__init__()
        self.features = inception_v3(pretrained=True, progress=False)

    @property
    def id(self):
        return None

    @property
    def out_dim(self):
        return 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input images.

        Returns:
            Embeddings.
        """

        x = self.features.Conv2d_1a_3x3(x)
        x = self.features.Conv2d_2a_3x3(x)
        x = self.features.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.features.Conv2d_3b_1x1(x)
        x = self.features.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.features.Mixed_5b(x)
        x = self.features.Mixed_5c(x)
        x = self.features.Mixed_5d(x)
        x = self.features.Mixed_6a(x)
        x = self.features.Mixed_6b(x)
        x = self.features.Mixed_6c(x)
        x = self.features.Mixed_6d(x)
        x = self.features.Mixed_6e(x)
        x = self.features.Mixed_7a(x)
        x = self.features.Mixed_7b(x)
        x = self.features.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # 2048
        return x


class DenseNetEmbeddings(Embeddings):
    """ResNet feature extractor.

    Features are extracted by Average Pooling layer of ResNets.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        See `Embeddings`.
        features: net doing the feature extraction.
        densenet_ids: possible DenseNet IDs (#layers).
        _densenet_out_dims: possible DenseNet output dimensions,
            same order as IDs.
        densenet_out_dim: mapping
    """

    densenet_ids = [121, 161, 169, 201]
    _densenet_out_dims = [1024, 2208, 1664, 1920]
    densenet_out_dim = {k: v for k, v in zip(densenet_ids, _densenet_out_dims)}
    densenet_out_dim.update(
        {str(k): v for k, v in zip(densenet_ids, _densenet_out_dims)}
    )

    def __init__(self, densenet_id: Optional[Union[int, str]] = 169):
        """Init.
        Args:
            densenet_id: ResNet version to load,
                default is 152 (i.e. ResNet152).
        """

        super().__init__()

        assert densenet_id in self.densenet_ids + list(
            map(str, self.densenet_ids)
        )

        self.densenet_id = densenet_id

        densenet = globals()["densenet{}".format(densenet_id)](
            pretrained=True, progress=False
        )

        self.features = densenet.features

    @property
    def id(self):
        return self.densenet_id

    @property
    def out_dim(self):
        return self.densenet_out_dim[self.densenet_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input images.

        Returns:
            Embeddings.
        """

        x = self.features(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return x


MODEL_MAPPING = {
    "densenet": DenseNetEmbeddings,
    "resnet": ResNetEmbeddings,
    "inception": Inception3Embeddings,
}
AVAILABLE_MODELS = list(MODEL_MAPPING)


def model_mapping(model: str) -> Type[Embeddings]:
    return MODEL_MAPPING[model]
