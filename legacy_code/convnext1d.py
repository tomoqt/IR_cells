import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (N, L, C) -> (N, C, L)

        x = input + self.drop_path(x)
        return x

class ConvNeXt1D(nn.Module):
    def __init__(self, in_chans=1, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.,
                 regression=False, regression_dim=2):
        super().__init__()

        self.regression = regression
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            Permute(),
            nn.LayerNorm(dims[0]),
            Permute()
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                Permute(),
                nn.LayerNorm(dims[i]),
                Permute(),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock1D(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1])
        
        if regression:
            self.head = nn.Linear(dims[-1], regression_dim)
        else:
            self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(-1))  # global average pooling, (N, C, L) -> (N, C)

    def forward(self, x):
        # Ensure input is 3D: (batch_size, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.forward_features(x)
        x = self.head(x)
        return x

class Permute(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.permute(0, 2, 1)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

def convnext_tiny_1d(num_classes=1000, regression=False, regression_dim=2):
    model = ConvNeXt1D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, regression=regression, regression_dim=regression_dim)
    return model

def convnext_small_1d(num_classes=1000, regression=False, regression_dim=2):
    model = ConvNeXt1D(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes, regression=regression, regression_dim=regression_dim)
    return model

def convnext_base_1d(num_classes=1000, regression=False, regression_dim=2):
    model = ConvNeXt1D(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, regression=regression, regression_dim=regression_dim)
    return model