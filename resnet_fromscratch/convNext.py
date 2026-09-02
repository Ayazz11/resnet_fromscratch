"""
ConvNeXt-Lite adapted for 32x32 inputs, trained from scratch (no pretrained weights).

Key adaptations vs. standard ConvNeXt (which assumes 224x224 + pretraining):
- Stem is a stride-1 3x3 conv instead of a stride-4 patchify — a stride-4 stem
  would collapse a 32x32 image to 8x8 before any features are learned.
- Downsampling happens between stages via 2x2 stride-2 convs (LayerNorm first,
  ConvNeXt-style), same total downsampling factor as your ResNet's maxpools:
  32 -> 32 -> 16 -> 8 -> 4.
- Channel widths and depths are scaled DOWN from the original ConvNeXt-Tiny
  (96/192/384/768, depths [3,3,9,3]) because you have no pretraining to lean
  on and likely a modest-sized dataset — an oversized model here will overfit
  faster than it generalizes. Tune `dims`/`depths` up if you have a lot of
  data and are underfitting.
- Includes LayerScale and stochastic depth (DropPath), which matter more than
  usual here since there's no pretraining to regularize the model implicitly.

Usage: import ConvNeXtLite and swap it in for CustomResNet(...) in your
training script — everything else (loss, mixup, optimizer, evaluation loop)
stays the same.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim for NCHW tensors (ConvNeXt-style)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (N, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_prob=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)  # pointwise, expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)  # pointwise, contract
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma[None, :, None, None] * x
        x = identity + self.drop_path(x)
        return x


class ConvNeXtLite(nn.Module):
    def __init__(self, num_classes=10, dims=(48, 96, 192, 384), depths=(2, 2, 4, 2),
                 drop_path_rate=0.1):
        super().__init__()

        # Gentle stem: stride 1, so 32x32 stays 32x32 before the first downsample.
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(dims[0]),
        )

        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_idx = 0

        for i in range(len(dims)):
            stage_blocks = nn.Sequential(*[
                ConvNeXtBlock(dims[i], drop_path_prob=dp_rates[block_idx + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            block_idx += depths[i]

            if i < len(dims) - 1:
                # Downsample between stages: 32->16->8->4 (3 downsamples for 4 stages)
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                ))
            else:
                self.downsamples.append(None)

        self.norm_out = LayerNorm2d(dims[-1])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)
            if downsample is not None:
                x = downsample(x)
        x = self.norm_out(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    # Quick shape/sanity check
    model = ConvNeXtLite(num_classes=10)
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", out.shape)
    print(f"Param count: {n_params:,}")