import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchgan.layers import MinibatchDiscrimination1d

from net_utils import weight_init


# %%
class Classifier(nn.Module):
    """
        A no-frills convolutional classifier.
        SELU activation mitigates the need for batch normalization layers.
    """

    def __init__(self, n_channels=3, n_classes=6):
        """
            Initialize the classifier.
        :param n_channels:
        :param n_classes: number of activities to classify
        """
        super(Classifier, self).__init__()
        self.c = nn.Sequential(
            nn.Conv2d(in_channels=n_channels,
                      out_channels=64,
                      kernel_size=(9, 1)),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1)),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1)),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Flatten(),

            nn.Linear(in_features=256 * 9 * 1, out_features=64),
            nn.SELU(),

            nn.Linear(in_features=64, out_features=n_classes),
        )
        self.apply(weight_init)

    def forward(self, x):
        x = self.c(x)
        x = F.log_softmax(x, dim=1)
        return x


# x = torch.rand((32, 3, 128, 1))
# net_c = Classifier(n_classes=6, n_channels=3)
# net_c(x).shape

# %%
class Discriminator(nn.Module):
    """
        Spectral Normalization used for the discriminator Conv layers
        https://arxiv.org/pdf/1805.08318.pdf

        Mini-batch Discrimination used in FC layers
    """

    def __init__(self, n_channels=3):
        """
            Initialize the discriminator.
        :param n_channels: number of IMU channels, 3 if using only the accelerometer
        """
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=(9, 1))),
            nn.SELU(),
            nn.AvgPool2d(kernel_size=(2, 1)),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1))),
            nn.SELU(),
            nn.AvgPool2d(kernel_size=(2, 1)),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1))),
            nn.SELU(),
            nn.AvgPool2d(kernel_size=(2, 1)),

            nn.Flatten(),

            nn.Linear(in_features=256 * 9 * 1, out_features=64),
            nn.SELU(),

            MinibatchDiscrimination1d(64, 32),
            nn.Linear(in_features=96, out_features=1),
            # nn.Linear(in_features=64, out_features=1),
        )
        self.apply(weight_init)

    def forward(self, x):
        x = self.d(x)
        return x


# x = torch.rand((32, 3, 128, 1))
# net_d = Discriminator(n_channels=3)
# net_d(x).shape


# %%
class KMaxPooling(nn.Module):
    """
        K-max pooling layer that performs an argmax over the k largest entries
        along the last dimension. For more details
        https://discuss.pytorch.org/t/resolved-how-to-implement-k-max-pooling-for-cnn-text-classification/931/4
    """

    def __init__(self, k, dim):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, x):
        index = x.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
        return x.gather(self.dim, index)


class LocalizationNetAvgPool(nn.Module):
    def __init__(self, window_size=128):
        """
            Initialize the localization network.
        :param window_size: the window length of the input samples, 128 by default
        """
        super(LocalizationNetAvgPool, self).__init__()

        n_conv1 = 64
        n_conv2 = 128

        self.net = nn.Sequential(  # input:  (batch, 1, window, channel)
            # spectral_norm(
            nn.Conv2d(in_channels=1, out_channels=n_conv1,
                      kernel_size=(1, 3)),  # output: (batch, 64, window, 1)
            nn.SELU(),
            # spectral_norm(
            nn.Conv2d(in_channels=n_conv1, out_channels=n_conv2,
                      kernel_size=(1, 1)),  # output: (batch, 128, window, 1)
            nn.SELU(),
            nn.AvgPool2d(kernel_size=(window_size,
                                      1)),  # output: (batch, 128, 1, 1)
            nn.Flatten())

    def forward(self, x): return self.net(x)


class LocalizationNetKMaxPool(nn.Module):
    """
        LocalizationNetKMaxPool is the convolutional localization network that uses K-max pooling
    """

    def __init__(self, window_size=128, n_pool=8):
        super(LocalizationNetKMaxPool, self).__init__()

        n_conv1 = 64
        n_conv2 = 128

        self.net = nn.Sequential(  # input:  (batch, 1, window, channel)
            # spectral_norm(
            nn.Conv2d(in_channels=1, out_channels=n_conv1,
                      kernel_size=(1, 3)),  # output: (batch, 64, window, 1)
            nn.SELU(),
            # spectral_norm(
            nn.Conv2d(in_channels=n_conv1, out_channels=n_conv2,
                      kernel_size=(1, 1)),  # output: (batch, 128, window, 1)
            nn.SELU(),
            KMaxPooling(n_pool, dim=2),  # output: (batch, 128, 4, 1)
            nn.Flatten())

    def forward(self, x):
        return self.net(x)


# x = torch.rand((32, 1, 128, 3))
# LocalizationNetKMaxPool()(x).shape
# %%

class SpatialTransformerBlock(nn.Module):
    """
    The main building block for the Generator, the Spatial Transformer Block uses the localization network in tandem with
    the fully connected network to regress the 12 (4x3) parameters for the affine transformation that needs to be applied to
    the input sample
    """

    def __init__(self, n_channel=3, window_size=128):
        super(SpatialTransformerBlock, self).__init__()

        self.n_channel = n_channel
        self.window_size = window_size

        self.n_conv1 = 64
        self.n_conv2 = 128
        self.n_fc1 = 64
        self.n_fc2 = 32
        self.n_pool = 8
        # self.n_pool = self.n_conv2 # when not doing any pooling,

        self.localization = LocalizationNetKMaxPool(window_size, self.n_pool)
        # self.localization = LocalizationNetAvgPool(window_size)

        self.fc_loc = nn.Sequential(  # input (batch, 128*8)
            nn.Linear(in_features=self.n_conv2 * self.n_pool,
                      out_features=self.n_fc1),  # output: (batch, 64)
            nn.SELU(),
            nn.Linear(in_features=self.n_fc1,
                      out_features=self.n_fc2),  # output: (batch, 32)
            nn.SELU(),
            nn.Linear(self.n_fc2, 12))

        self.apply(weight_init)

        # ----------------------------------------------------------------------
        # need to make final fc layers weights zero and biases as identity
        # wrong initialization will lead to non convergence
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]))
        # ---------------------------------------------------------------------

    def forward(self, x):  # input:  (batch, channel, window, 1)
        x = x.permute(0, 3, 2, 1)  # output: (batch, 1, window, channel)

        xs = self.localization(x)  # output: (batch, 128*8, 1)
        theta = self.fc_loc(xs)  # output: (batch, 12)
        theta = theta.view(-1, 4, 3)  # output: (batch, 4, 3)

        x = x.squeeze(1)  # output: (batch, window, channel)
        ones = torch.ones(x.shape[0],
                          x.shape[1],
                          1,
                          dtype=torch.float,
                          requires_grad=False).to(x.device)
        aug = torch.cat([x, ones], 2)  # output: (batch, window, channel+1)

        x = torch.matmul(aug, theta)  # output: (batch, window, channel)
        x = x.unsqueeze(1)  # output: (batch, 1, window, channel)

        return x.permute(0, 3, 2, 1).contiguous(), theta  # output: (batch, channel, window, 1)


class Generator(nn.Module):
    """
    Wrapper around the Spatial Transformer Block, in case we want to stack more of them,
    stacking leads to better results at the expense of more parameters and training/inference time.
    """

    def __init__(self, n_blocks=2, n_channels=3, window_size=128):
        super(Generator, self).__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.Sequential(
            *[SpatialTransformerBlock(n_channels, window_size) for i in range(n_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)

# x = torch.rand((32, 3, 128, 1))
# g = Generator(2, 3, 128)
# g(x).shape

# %%
