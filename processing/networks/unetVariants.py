import torch.nn as nn
from torch import cat, tensor

from processing.networks.unet import UNet

class UNetHalfPad2(UNet):
    def __init__(self, in_channels:int=2, out_channels:int=1, init_features:int=32, depth:int=3, kernel_size:int=5, activation:str="relu", norm:str="batchnorm"):
        super().__init__()
        features = init_features
        activation = get_activation_fct(activation)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, activation=activation, norm=norm))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, activation=activation, norm=norm))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(self._block(features, features//2, kernel_size=kernel_size, activation=activation, norm=norm))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            required_size = x.shape[2]
            start_pos = (encoding.shape[2] - required_size)//2
            encoding = encoding[:, :, start_pos:start_pos+required_size, :]
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, activation=nn.ReLU(inplace=True), norm="batchnorm"):
        direction = "horizontal"
        return nn.Sequential(
            OneSidePadding(kernel_size, direction),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),
            _build_norm2d(features, norm),
            activation,
        )
    
        
class UNetNoPad2(UNet):
    def __init__(self, in_channels:int=2, out_channels:int=1, init_features:int=32, depth:int=3, kernel_size:int=5, stride:int=1, dilation:int=1, activation:str="relu", norm:str="batchnorm", repeat_inner:bool=False):
        super().__init__()
        features = init_features
        activation = get_activation_fct(activation)
        self.stride = stride

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        for _ in range(depth):
            self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, stride=stride, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, stride=stride, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(self._block(features, features//2, kernel_size=kernel_size, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            required_size = x.shape[2:]
            start_pos = ((encoding.shape[2] - required_size[0])//2, (encoding.shape[3] - required_size[1])//2)
            encoding = encoding[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, stride=1, dilation=1, activation=nn.ReLU(inplace=True), norm:str=None, repeat_inner=False):    
        if repeat_inner:
            return nn.Sequential(
                _build_conv2d(in_channels, features, kernel_size, stride, dilation),
                _build_norm2d(features, norm),
                activation,
                _build_conv2d(features, features, kernel_size, stride, dilation),
                activation,
            )
        else:
            return nn.Sequential(
                _build_conv2d(in_channels, features, kernel_size, stride, dilation),
                _build_norm2d(features, norm),
                activation,
            )

class OneSidePadding(nn.Module):
    def __init__(self, kernel_size, direction):
        super().__init__()
        self.pad_len = kernel_size//2
        self.direction = direction

    def forward(self, x:tensor) -> tensor:
        if self.direction == "vertical":
            padding_vector = (0,)*2 + (self.pad_len,)*2  
            result = nn.functional.pad(x, padding_vector, mode='constant')     # or 'reflect'?
        elif self.direction == "horizontal":
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(x, padding_vector, mode='constant')
        return result
    
class PaddingCircular(nn.Module):
    def __init__(self, kernel_size, direction="both"):
        super().__init__()
        self.pad_len = kernel_size//2
        self.direction = direction

    def forward(self, x:tensor) -> tensor:
        if self.direction == "both":
            padding_vector = (self.pad_len,)*4
            result = nn.functional.pad(x, padding_vector, mode='circular')    

        elif self.direction == "horizontal":
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(x, padding_vector, mode='circular')    
            padding_vector = (0,)*2 + (self.pad_len,)*2  
            result = nn.functional.pad(result, padding_vector, mode='constant')     # or 'reflect'?

        elif self.direction == "vertical":
            padding_vector = (0,)*2 + (self.pad_len,)*2
            result = nn.functional.pad(x, padding_vector, mode='circular')    
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(result, padding_vector, mode='constant')

        elif self.direction == "none":
            padding_vector = (self.pad_len,)*4
            result = nn.functional.pad(x, padding_vector, mode='constant')    
        return result
    
def get_activation_fct(name:str):
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif name.lower() == "tanh":
        return nn.Tanh()

@staticmethod
def _build_conv2d(in_channels, features, kernel_size, stride, dilation):
    return nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding="valid", # "valid":= no padding, "same":=padding (default is with 0)
                bias=True,
            )

@staticmethod
def _build_norm2d(features, norm):
    if norm.lower() == "batchnorm":
        return nn.BatchNorm2d(num_features=features)
    elif norm.lower() == "groupnorm":
        return nn.GroupNorm(num_groups=4, num_channels=features)
    else:
        return nn.Identity()

    