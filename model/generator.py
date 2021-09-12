# IMPORTS
import torch
import torch.nn as nn

# variables
image_size = (3, 128, 128)

# Weight initialization for discriminator layers
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

'''
WGAN Generator class. Ready for generating either batch_sizexchannelx64x64
    or batch_sizexchannelx128x128. images.
PARAMETERS
'input size' : random noise vector dimension.
'output channels': To determine image is RGB or Grayscale.
'nf': Generator Filters. default : 128
'''

class Generator(nn.Module):
    def __init__(self, input_size, output_channels, nf=128):
        super(Generator, self).__init__()

        if image_size[1] == 64:
            self.first_block = nn.Sequential(
                nn.ConvTranspose2d(input_size, nf*8, 4, stride=1,
                                padding=0, bias=False),
                nn.BatchNorm2d(nf*8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        elif image_size[1] == 128:
            self.first_block = nn.Sequential(
                nn.ConvTranspose2d(input_size, nf*16, 4, stride=1,
                                padding=0, bias=False),
                nn.BatchNorm2d(nf*16),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ConvTranspose2d(nf*16, nf*8, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf*8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(nf*8, nf*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(nf, output_channels, 4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.first_block(x)
        x = self.conv_block(x)
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

