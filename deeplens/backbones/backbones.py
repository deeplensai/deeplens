import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool=True, pool_size=2, pool_stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if isinstance(pool, bool):
            if pool:
                self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        elif isinstance(pool, str):
            if pool == 'avg':
                self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)
            else:
                self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x) if self.pool else x
        return x