import torch


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class StudentBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, act = torch.nn.Hardswish):
        super(StudentBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = act()

    def forward(self, x):
        return self.relu(self.in1(self.conv1(x)))

# mods = [1,0,0,2,0,0] #2
# mods = [3,8,13,12,4,0] #26
mods = [3,14,15,14,4,0] #36
# mods = [3,15,15,15,13,1] #46
# mods = [8,20,20,20,19,3] #74
# udnie
# mods = [8,19,19,13,6,0] #34
inner_channels = 24
class Student(torch.nn.Module):
    def __init__(self, modifications = [0]*6, first_kernel_size = 9):
        super(Student, self).__init__()
        # Initial convolution layers
        i = 0
        self.conv1 = ConvLayer(3, inner_channels // 2 - modifications[0], kernel_size=first_kernel_size, stride=2)
        self.in1 = torch.nn.InstanceNorm2d(inner_channels // 2 - modifications[0], affine=True)
        self.conv2 = ConvLayer(inner_channels // 2 - modifications[0], inner_channels - modifications[1], kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(inner_channels - modifications[1], affine=True)
        # Residual layers
        self.res1 = StudentBlock(inner_channels - modifications[1], inner_channels - modifications[2], kernel_size=3)
        self.res2 = StudentBlock(inner_channels - modifications[2], inner_channels - modifications[3], kernel_size=3)
        self.res3 = StudentBlock(inner_channels - modifications[3], inner_channels - modifications[4], kernel_size=3)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(inner_channels - modifications[4], inner_channels // 2 - modifications[5], kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(inner_channels // 2 - modifications[5], affine=True)
        self.deconv2 = UpsampleConvLayer(inner_channels // 2 - modifications[5], 3, kernel_size=first_kernel_size, stride=1, upsample=2)
        # Non-linearities
        self.relu = torch.nn.LeakyReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.deconv2(y)
        return y

