
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
WGAN Discriminator class.
PARAMETERS
'input channels': To determine image is RGB or Grayscale.
'nf': Discriminator Filters
'''

class Discriminator(nn.Module):
    def __init__(self, input_channels, nf):
        super(Discriminator, self).__init__()
        self.flattened_size = 64 * \
            (image_size[1]//2//2//2) * (image_size[2]//2//2//2)
        self.conv_block = nn.Sequential(
            # input is (3, 32, 32)
            nn.Conv2d(input_channels, nf, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf, 16, 16)
            nn.Conv2d(nf, nf * 2, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*2, 8, 8)
            nn.Conv2d(nf * 2, nf * 4, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(nf * 4, nf * 8, 4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # input is (nf*4, 4, 4)
            nn.Conv2d(nf * 8, 1, 4, padding=0, stride=1, bias=False),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x.view(-1, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
