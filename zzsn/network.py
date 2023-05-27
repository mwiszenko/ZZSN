from torch import nn


class ProtoNetwork(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim):
        super(ProtoNetwork).__init__()
        self.encoder = nn.Sequential(
            get_conv_block(x_dim[0], hid_dim),
            get_conv_block(hid_dim, hid_dim),
            get_conv_block(hid_dim, hid_dim),
            get_conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        return x.view(x.size(0), -1)


def get_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
