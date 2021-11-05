"""VGG11/13/16/19 in Pytorch."""
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator_upsample_ver(nn.Module):
    def __init__(self, n_classes, img_size, latent_dim, channels=1):
        super(Generator_upsample_ver, self).__init__()

        self.init_size = img_size // 2
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.l1 = nn.Sequential(nn.Linear(latent_dim + n_classes, 64 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        out = self.l1(gen_input)

        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        # img = self.conv_blocks(out)

        out = self.conv_trans1(out)
        out = self.bn1(out)
        out = self.leaky_relu1(out)
        out = self.conv_trans2(out)
        img = self.sigmoid(out)

        return img

# ===================================
#           AC GAN
# ============= start ===============
# Generator Code
class GNetBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(GNetBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

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

        self.convt1 = GNetBlock(ngf * 8, ngf * 4, 0)  # 8*8
        self.convt2 = GNetBlock(ngf * 4, ngf * 2, 0)  # 16*16
        self.convt3 = GNetBlock(ngf * 2, ngf, 0)      # 32 *32

        self.output = nn.Sequential(
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            # nn.ConvTranspose2d(ngf, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        '''
        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.5),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.5),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.5),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            # nn.ConvTranspose2d(ngf, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        '''
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
    def __init__(self, in_size, out_size, dropout=0.0, bn=True):
        super(DNetBlock, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_size))

        layers.append(nn.LeakyReLU(0.01, True))

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

        # input 3*64*64
        self.layer1 = DNetBlock(channels, d, 0.5)

        # input 64*32*32
        self.layer2 = DNetBlock(d, d*2, 0.5)

        # input 128*16*16
        self.layer3 = DNetBlock(d*2, d*4, 0.5)

        # input 256*8*8
        self.layer4 = DNetBlock(d*4, d*8, 0.5)

        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(d*8, 1, 4, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(d*8, n_classes, 4, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

        # input 512*2*2 用于输入尺寸是32*32
        # self.validity_layer = nn.Sequential(nn.Conv2d(d*8, 1, 2, 1, 0, bias=False),
        #                                     nn.Sigmoid())
        #
        # self.label_layer = nn.Sequential(nn.Conv2d(d*8, n_classes, 2, 1, 0, bias=False),
        #                                  nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, self.n_classes)

        return validity, plabel


class DNetClassifier(nn.Module):
    def __init__(self, n_classes, channels=1, d=32):
        super(DNetClassifier, self).__init__()

        self.n_classes = n_classes
        # input 3*64*64
        # input 3*64*64
        self.layer1 = DNetBlock(channels, d, 0.5)

        # input 64*32*32
        self.layer2 = DNetBlock(d, d*2, 0.5)

        # input 128*16*16
        self.layer3 = DNetBlock(d*2, d*4, 0.5)

        # input 256*8*8
        self.layer4 = DNetBlock(d*4, d*8, 0.5)
        # 用于尺寸是64*64
        self.classifier = nn.Sequential(nn.Conv2d(d*8, n_classes, 4, 1, 0, bias=False))
        # 用于尺寸是32*32
        # self.classifier = nn.Sequential(nn.Conv2d(d * 8, n_classes, 2, 1, 0, bias=False))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        plabel = self.classifier(x)
        plabel = plabel.view(-1, self.n_classes)

        return plabel


# ============== end ================


# G(z)
class ConvTranGenerator(nn.Module):
    # initializers
    def __init__(self, n_classes, latent_dim, channels=1, d=128, bn=True):
        super(ConvTranGenerator, self).__init__()
        self.if_bn = bn
        self.deconv1_1 = nn.ConvTranspose2d(latent_dim, d * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d * 2)
        self.deconv1_2 = nn.ConvTranspose2d(n_classes, d * 2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z, label):
        # gen_input = torch.cat((self.label_emb(label), z), -1)     # 先cat后带入卷积
        if self.if_bn:
            x = self.deconv1_1_bn(self.deconv1_1(z))
        else:
            x = self.deconv1_1(z)
        x = F.relu(x, True)
        if self.if_bn:
            y = self.deconv1_2_bn(self.deconv1_2(label))
        else:
            y = self.deconv1_2(label)
        y = F.relu(y, True)
        x = torch.cat([x, y], 1)
        if self.if_bn:
            x = self.deconv2_bn(self.deconv2(x))
        else:
            x = self.deconv2(x)
        x = F.relu(x, True)
        if self.if_bn:
            x = self.deconv3_bn(self.deconv3(x))
        else:
            x = self.deconv3(x)
        x = F.relu(x, True)
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x


class ConvTranDiscriminator(nn.Module):
    # initializers
    def __init__(self, n_classes, channels, d=128):
        super(ConvTranDiscriminator, self).__init__()
        self.conv1_input = nn.Conv2d(channels, d // 2, 4, 2, 0)
        self.conv1_label = nn.Conv2d(n_classes, d // 2, 4, 2, 0)
        self.conv2 = nn.Conv2d(d, d * 2, 3, 2, 0)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 3, 1, 0)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 1, 1)
        self.l1 = nn.Linear(25, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_input(input), 0.2)
        y = F.leaky_relu(self.conv1_label(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = torch.sigmoid(self.conv4(x))
        x = self.conv4(x)
        x = x.view(input.shape[0], -1)
        x = F.relu(x)
        x = self.l1(x)
        # x = torch.sigmoid(x)

        return x


class MyIncepResDiscriminator(nn.Module):
    def __init__(self, n_classes=6, in_channels=3, d=64):
        super(MyIncepResDiscriminator, self).__init__()

        self.n_classes = n_classes

        self.stem = SimpleStem(in_channels)
        # self.conv1_label = nn.Conv2d(n_classes, d, 3, 1, 0)

        self.incepres_in = my_IncepResNet_unit(160, d * 2, 1)
        self.incepres_2 = my_IncepResNet_unit(d * 2, d*2, 1)
        self.incepres_3 = my_IncepResNet_unit(d*2, d, 1)
        self.incepres_4 = my_IncepResNet_unit(d, d, 1)

        self.validity_layer = nn.Sequential(nn.Conv2d(d, 1, 3, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(d, n_classes, 3, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # def forward(self, x, label):
    #     x = self.stem(x)
    #     y = F.leaky_relu(self.conv1_label(label), 0.2)
    #     x = torch.cat([x, y], 1)
    #     x = self.incepres_in(x)
    #     x = self.incepres_2(x)
    #     x = self.incepres_3(x)
    #     x = self.conv(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fcn(x)
    #     return x

    def forward(self, x):
        x = self.stem(x)
        x = self.incepres_in(x)
        x = self.incepres_2(x)
        x = self.incepres_3(x)

        x = self.incepres_4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, self.n_classes)

        return validity, plabel


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DNetClassifier_old(nn.Module):
    # 与GAN中判别器构造相同的分类器
    def __init__(self, n_classes, channels, d=128):
        super(DNetClassifier_old, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d, 4, 2, 0)
        self.conv2 = nn.Conv2d(d, d * 2, 3, 2, 0)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 3, 1, 0)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 1, 1)
        self.fc = nn.Linear(25, n_classes)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = torch.sigmoid(self.conv4(x))
        x = self.conv4(x)
        x = x.view(input.shape[0], -1)
        x = F.relu(x)
        x = self.fc(x)
        # x = torch.sigmoid(x)

        return x


# ============
# MyNet
# ============
class MySimpleNet(nn.Module):
    def __init__(self, classes=6, in_channels=3, d_filters=64):
        super(MySimpleNet, self).__init__()

        self.stem = SimpleStem(in_channels)
        self.incepres_1 = my_IncepResNet_unit(160, d_filters * 2, 1)
        self.incepres_2 = my_IncepResNet_unit(d_filters * 2, d_filters, 1)
        self.incepres_3 = my_IncepResNet_unit(d_filters, 64, 1)

        self.conv = Conv2d(64, 256, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incepres_1(x)
        x = self.incepres_2(x)
        x = self.incepres_3(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
        self.branch_0 = Conv2d(in_channels, br_channel, 2, stride=2, padding=0, bias=False)

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
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, br_channel, 3, stride=1, padding=1, bias=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x = self.maxpool(x)
        x0 = self.branch_0(x)
        # x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        # x_res = torch.cat((x1, x2), dim=1)
        # x = self.conv(x)
        return self.prelu(x0 + self.scale * x2)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        # self.gn = nn.GroupNorm(32, out_channels)
        # self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.gn(x)
        # x = self.relu(x)
        x = self.lrelu(x)
        return x


# 简单的分类网络
class Classifier(nn.Module):
    def __init__(self, n_classes, img_size, channels=1):
        super(Classifier, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.classifier = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes))

    def forward(self, img):
        x = self.model(img)
        x = x.view(x.size(0), -1)
        c = self.classifier(x)
        return c


class SimpleCNN(nn.Module):
    def __init__(self, out_dim=640):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.Softmax(dim=1)
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.fc1(x))
        x = self.classifier(x)
        return x


# 用于local model的模型
class VGG_Models(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_Models, self).__init__()
        if isinstance(vgg_name, int):
            vgg_name = str(vgg_name)
        cfg = {
            '0': [16, 'M', 32, 'M', 64, 'M'],
            '1': [16, 16, 'M', 32, 'M', 64, 'M'],
            '2': [16, 16, 'M', 32, 32, 'M', 64, 'M'],
            '3': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M'],
            'VGGs': [32, 'M', 64, 64, 'M', 64, 'M'],
            'VGGs3': [32, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
            'VGGs2': [32, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M'],
            'VGGs1': [32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M'],
        }
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        # out = F.dropout(out, p=0.5, training=self.training)
        # out = self.classifier(out)
        # out = self.softmax(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.LeakyReLU(0.2, inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
        return nn.Sequential(*layers)


cfg = {
    'VGGs': [32, 'M', 64, 64, 'M', 64, 'M'],
    'VGGs3': [32, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGGs2': [32, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGGs1': [32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, channels, n_classes):
        super(VGG, self).__init__()
        self.in_channels = channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
