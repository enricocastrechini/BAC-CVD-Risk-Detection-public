import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.stochastic_depth(input, self.p, self.mode, self.training)

    def stochastic_depth(self, input: torch.Tensor, p: float, mode: str, training: bool) -> torch.Tensor:
        if p == 0.0 or not training:
            return input

        if mode == "row":
            batch_size = input.shape[0]
            random_tensor = (1 - p) + torch.rand([batch_size, 1, 1, 1], dtype=input.dtype, device=input.device)
            output = input / (1 - p) * random_tensor
        else:
            raise ValueError(f"Unsupported mode {mode}")
        return output

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class CNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

class CNBlockConfig:
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1,  # Change to 1 for binary classification
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, firstconv_output_channels, kernel_size=4, stride=4, padding=0),
                norm_layer(firstconv_output_channels),
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)  # Output 1 value for binary classification
        # )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lastconv_output_channels),  # Change to LayerNorm
            nn.Flatten(1),                           # Adjust index to (1)
            nn.Linear(lastconv_output_channels, num_classes)  # Adjust index to (2)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_feat_modules(self):
        return self.features

    def get_bn_before_relu(self):
        bn_layers = []
        for layer in self.features:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, CNBlock):
                        bn_layers.append(sub_layer.block[2])
        return bn_layers

    def _forward_impl(self, x: torch.Tensor, is_feat: bool = False, preact: bool = False):
        features = []
        preact_features = []

        x = self.features[0](x)
        if is_feat:
            features.append(x)
        for i in range(1, len(self.features)):
            x = self.features[i](x)
            if is_feat:
                features.append(x)
            if preact:
                preact_features.append(x)

        x = self.avgpool(x)
        if is_feat:
            features.append(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if is_feat:
            if preact:
                return preact_features, x
            else:
                return features, x
        else:
            return x

    def forward(self, x: torch.Tensor, is_feat: bool = False, preact: bool = False) -> torch.Tensor:
        return self._forward_impl(x, is_feat=is_feat, preact=preact)

def convnext_tiny(**kwargs: Any) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    return ConvNeXt(block_setting, stochastic_depth_prob=0.1, **kwargs)

def convnext_small(**kwargs: Any) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    return ConvNeXt(block_setting, stochastic_depth_prob=0.4, **kwargs)

def convnext_base(**kwargs: Any) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    return ConvNeXt(block_setting, stochastic_depth_prob=0.5, **kwargs)

def convnext_large(**kwargs: Any) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    return ConvNeXt(block_setting, stochastic_depth_prob=0.5, **kwargs)

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = convnext_small(num_classes=1)  # Change num_classes to 1
    print(model)
    feats, logit = model(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in model.get_bn_before_relu():
        if isinstance(m, nn.LayerNorm):
            print('pass')
        else:
            print('warning')