import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================
#           AC GAN
# ============= start ===============
# Generator Code
class GNetBlock(nn.Module):
    def __init__(self, in_size, out_size, norm='gn', dropout=0.0):
        super(GNetBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)
        ]
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_size))
        if norm == 'gn':
            layers.append(nn.GroupNorm(32, out_size))
        if dropout:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.ReLU(True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, channels=1, d=32):
        super(Generator, self).__init__()
        ngf = d
        self.latent_dim = latent_dim
        self.n_c = n_classes
        self.convt_input_dim = latent_dim + n_classes

        self.input_z = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim + n_classes, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # nn.LeakyReLU(0.2, True),
        )

        self.embedding = nn.Embedding(n_classes, n_classes)
        # self.input_label = nn.Sequential(
        #     # input is label, going into a convolution
        #     nn.ConvTranspose2d(n_classes, ngf * 1, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 1),
        #     nn.ReLU(True)
        # )

        bn = True
        drop = 0
        norm = 'gn'

        self.convt1 = GNetBlock(ngf * 8, ngf * 4, norm, drop)  # 8*8
        self.convt2 = GNetBlock(ngf * 4, ngf * 2, norm, drop)  # 16*16
        self.convt3 = GNetBlock(ngf * 2, ngf, norm, drop)      # 32 *32

        self.output = nn.Sequential(
            nn.ConvTranspose2d(ngf, channels, 1, 1, 0, bias=False),
            # nn.ConvTranspose2d(ngf, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        # x = self.input_z(input)
        # y = self.input_label(label)
        # inputs = torch.cat([x, y], 1)

        lb_embedding = self.embedding(label)
        # x = torch.mul(input, lb_embedding)
        # x = x.view(-1, self.latent_dim, 1, 1)

        x = torch.cat((input, lb_embedding), -1)
        x = x.view(-1, self.convt_input_dim, 1, 1)

        x = self.input_z(x)

        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.output(x)

        return x


class DNetBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, norm='gn', activate_func='leakyrelu'):
        super(DNetBlock, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
        ]

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_size))
        if norm == 'gn':
            layers.append(nn.GroupNorm(32, out_size))

        if activate_func == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2, True))
        else:
            layers.append(nn.ReLU(True))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, n_classes, channels=1, d=32):
        super(Discriminator, self).__init__()

        self.n_classes = n_classes
        self.embedding = nn.Embedding(n_classes, n_classes)

        norm = 'gn'
        # input 3*64*64
        self.layer1 = DNetBlock(channels, d, 0.5)

        # input 64*32*32
        self.layer2 = DNetBlock(d, d*2, 0.5)

        # input 128*16*16
        self.layer3 = DNetBlock(d*2, d*4, 0.5)

        # input 256*8*8
        self.layer4 = DNetBlock(d*4, d*8, 0.5)

        # input 512*2*2 用于输入尺寸是32*32
        self.validity_layer = nn.Sequential(nn.Conv2d(d*8, 1, 2, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.validity_fcn_layer = nn.Sequential(nn.Linear(1024+n_classes, 1),
                                                nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(d*8, n_classes, 2, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x, y=None):
        if y is not None:
            lb_embedding = self.embedding(y)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if y is None:
            validity = self.validity_layer(x)
            plabel = self.label_layer(x)

            validity = validity.view(-1)
            plabel = plabel.view(-1, self.n_classes)

            return validity, plabel
        else:
            x = torch.flatten(x, start_dim=1)

            x = torch.cat((x, lb_embedding), -1)
            validity = self.validity_fcn_layer(x)
            validity = validity.view(-1)
            return validity



class DNetClassifier(nn.Module):
    def __init__(self, n_classes, channels=1, d=32):
        super(DNetClassifier, self).__init__()

        self.n_classes = n_classes

        dropout = 0.5
        norm = 'bn'
        act_f = 'relu'

        # input 3*64*64
        self.layer1 = DNetBlock(channels, d, dropout, norm, act_f)

        # input 64*32*32
        self.layer2 = DNetBlock(d, d*2, dropout, norm, act_f)

        # input 128*16*16
        self.layer3 = DNetBlock(d*2, d*4, dropout, norm, act_f)

        # input 256*8*8
        self.layer4 = DNetBlock(d*4, d*8, dropout, norm, act_f)

        # 用于尺寸是64*64
        # self.classifier = nn.Sequential(nn.Conv2d(d*8, n_classes, 4, 1, 0, bias=False))
        # 用于尺寸是32*32
        self.classifier = nn.Linear(4*8*d, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        plabel = self.classifier(x)
        # plabel = plabel.view(-1, self.n_classes)

        return plabel


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class MySimpleNet(nn.Module):
    def __init__(self, in_channels=3, classes=6):
        super(MySimpleNet, self).__init__()

        self.stem = SimpleStem(in_channels)
        self.incepres_1 = my_IncepResNet_unit(160, 64, 1)
        self.incepres_2 = my_IncepResNet_unit(64, 64, 1)
        self.incepres_3 = my_IncepResNet_unit(64, 64, 1)

        self.conv = Conv2d(64, 256, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incepres_1(x)
        x = self.incepres_2(x)
        x = self.incepres_3(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=1, padding=0, bias=False),  # 149 x 149 x 32
            # Conv2d(32, 32, 3, stride=1, padding=0, bias=False),  # 147 x 147 x 32
            # Conv2d(32, 32, 3, stride=1, padding=1, bias=False),  # 147 x 147 x 64
            # nn.MaxPool2d(2, stride=1, padding=0),  # 73 x 73 x 64
        )

        self.branch_0 = Conv2d(32, 32, 1, stride=1, padding=0, bias=False)

        self.branch_1 = nn.Sequential(
            Conv2d(32, 48, 1, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(32, 48, 1, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=1, bias=False),
            Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(32, 64, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class SimpleStem(nn.Module):
    def __init__(self, in_channels):
        super(SimpleStem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=1, padding=0, bias=False),  # 149 x 149 x 32
            # Conv2d(32, 32, 3, stride=1, padding=0, bias=False),  # 147 x 147 x 32
            # Conv2d(32, 32, 3, stride=1, padding=1, bias=False),  # 147 x 147 x 64
            # nn.MaxPool2d(2, stride=1, padding=0),  # 73 x 73 x 64
        )

        self.branch_0 = Conv2d(32, 32, 1, stride=1, padding=0, bias=False)

        self.branch_1 = nn.Sequential(
            Conv2d(32, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(32, 32, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class my_IncepResNet_unit(nn.Module):
    def __init__(self, in_channels, br_channel=32, scale=1.0):
        super(my_IncepResNet_unit, self).__init__()
        self.scale = scale

        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)

        # self.branch_1 = nn.Sequential(
        #     Conv2d(in_channels, br_channel, 1, stride=1, padding=0, bias=False),
        #     Conv2d(br_channel, br_channel*2, 3, stride=1, padding=1, bias=False)
        # )

        self.branch_2 = nn.Sequential(
            # Conv2d(in_channels, br_channel, 1, stride=1, padding=0, bias=False),
            Conv2d(in_channels, br_channel, 3, stride=1, padding=1, bias=False),
            Conv2d(br_channel, br_channel, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, br_channel, 3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x = self.maxpool(x)
        # x0 = self.branch_0(x)
        # x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        # x_res = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        return self.prelu(x + self.scale * x2)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.gn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.gn(x)
        x = self.relu(x)
        return x