import torch.nn as nn
from .unet import UNet

class UNet3d(UNet):
    def __init__(self, in_channels=4, out_channels=1, init_features=16, depth= 3, kernel_size=5):
        super().__init__(in_channels, out_channels, init_features, depth, kernel_size)
        features = init_features
        padding_mode =  "zeros"            
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(UNet3d._block(in_channels, features, kernel_size=kernel_size, padding_mode=padding_mode))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(UNet3d._block(in_channels, features, kernel_size=kernel_size, padding_mode=padding_mode))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose3d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(UNet3d._block(features, features//2, kernel_size=kernel_size, padding_mode=padding_mode))
            features = features // 2

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    
    @staticmethod
    def _block(in_channels, features, kernel_size=5, padding_mode="zeros"):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                #padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                #padding_mode=padding_mode,
                bias=True,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                #padding_mode=padding_mode,
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()

