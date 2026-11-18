import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm="instance", residual=True):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            self._norm2d(norm, out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            self._norm2d(norm, out_channels),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            self.shortcut = nn.Identity()

    @staticmethod
    def _norm2d(norm, channels):
        if norm in ("instance", "IN"):
            return nn.InstanceNorm2d(channels)
        if norm in ("batch", "BN"):
            return nn.BatchNorm2d(channels)
        if norm in (None, "none", "None"):
            return nn.Identity()
        raise ValueError(f"Unsupported normalization: {norm}")

    def forward(self, x):
        y = self.conv(x)
        if self.residual:
            y = y + self.shortcut(x)
        return self.act(y)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        features=(16, 32, 64, 128),
        encoder_norm="instance",
        decoder_norm=None,
        residual=True,
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for idx, feat in enumerate(features):
            self.encoder.append(
                DoubleConv2D(
                    in_channels if idx == 0 else features[idx - 1],
                    feat,
                    norm=encoder_norm,
                    residual=residual,
                )
            )

        self.bottleneck = DoubleConv2D(
            features[-1], features[-1] * 2, norm=encoder_norm, residual=residual
        )

        reversed_features = list(reversed(features))
        for idx, feat in enumerate(reversed_features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feat * 2 if idx == 0 else reversed_features[idx - 1],
                    feat,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoder.append(
                DoubleConv2D(feat * 2, feat, norm=decoder_norm, residual=residual)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = reversed(skips)
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = next(skips)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx + 1](x)
        return self.final_conv(x)


class SeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DoubleConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm="instance",
        residual=True,
        kernel_size=(3, 3, 3),
        padding=(1, 1, 1),
        use_separable=False,
    ):
        super().__init__()
        self.residual = residual
        conv = SeparableConv3D if use_separable else nn.Conv3d
        self.conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
            if conv is nn.Conv3d
            else conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            self._norm3d(norm, out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
            if conv is nn.Conv3d
            else conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            self._norm3d(norm, out_channels),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            self.shortcut = nn.Identity()

    @staticmethod
    def _norm3d(norm, channels):
        if norm in ("instance", "IN"):
            return nn.InstanceNorm3d(channels)
        if norm in ("batch", "BN"):
            return nn.BatchNorm3d(channels)
        if norm in (None, "none", "None"):
            return nn.Identity()
        raise ValueError(f"Unsupported normalization: {norm}")

    def forward(self, x):
        y = self.conv(x)
        if self.residual:
            y = y + self.shortcut(x)
        return self.act(y)


class DeterministicMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        if stride is None:
            self.sd, self.sh, self.sw = self.kd, self.kh, self.kw
        elif isinstance(stride, int):
            self.sd = self.sh = self.sw = stride
        else:
            self.sd, self.sh, self.sw = stride
        if isinstance(padding, int):
            self.pad_d = self.pad_h = self.pad_w = padding
        else:
            self.pad_d, self.pad_h, self.pad_w = padding

    def forward(self, x):
        n, c, d, h, w = x.shape
        x = F.pad(
            x,
            (self.pad_w, self.pad_w, self.pad_h, self.pad_h, self.pad_d, self.pad_d),
            mode="constant",
            value=float("-inf"),
        )
        x = x.unfold(2, self.kd, self.sd).unfold(3, self.kh, self.sh).unfold(4, self.kw, self.sw)
        x = x.contiguous().view(n, c, x.shape[2], x.shape[3], x.shape[4], -1)
        out, _ = x.max(dim=-1)
        return out


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        features=(16, 32, 64, 128),
        encoder_norm="instance",
        decoder_norm=None,
        residual=True,
        kernel_size=(3, 3, 3),
        padding=(1, 1, 1),
        use_separable=False,
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        pools = []

        for idx, feat in enumerate(features):
            self.encoder.append(
                DoubleConv3D(
                    in_channels if idx == 0 else features[idx - 1],
                    feat,
                    norm=encoder_norm,
                    residual=residual,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_separable=use_separable,
                )
            )
            if idx % 2 == 0:
                pools.append(DeterministicMaxPool3d(kernel_size=2, stride=2))
            else:
                pools.append(DeterministicMaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.pool = nn.ModuleList(pools)

        self.bottleneck = DoubleConv3D(
            features[-1],
            features[-1] * 2,
            norm=encoder_norm,
            residual=residual,
            kernel_size=kernel_size,
            padding=padding,
            use_separable=use_separable,
        )

        reversed_features = list(reversed(features))
        for feat_in, feat_out, pool in zip(
            [features[-1] * 2] + reversed_features[:-1], reversed_features, reversed(self.pool)
        ):
            scale = (pool.sd, pool.sh, pool.sw)
            self.upconvs.append(
                nn.Sequential(nn.ConvTranspose3d(feat_in, feat_out, kernel_size=scale, stride=scale))
            )
            self.decoder.append(
                DoubleConv3D(
                    feat_out * 2,
                    feat_out,
                    norm=decoder_norm,
                    residual=residual,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_separable=use_separable,
                )
            )

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for block, pool in zip(self.encoder, self.pool):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = reversed(skips)
        for up, block in zip(self.upconvs, self.decoder):
            x = up(x)
            skip = next(skips)
            dz = skip.shape[2] - x.shape[2]
            dy = skip.shape[3] - x.shape[3]
            dx = skip.shape[4] - x.shape[4]
            if dz < 0 or dy < 0 or dx < 0:
                raise RuntimeError("Unexpected decoder shape mismatch.")
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
            x = torch.cat((skip, x), dim=1)
            x = block(x)
        return self.final_conv(x)
