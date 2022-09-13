from collections import OrderedDict
from torch import nn
import torch
import torch.nn.functional as F


class resnet_nonlocal(nn.Module):
    def __init__(self, blocks=2, input_nc=3, feature_dim=784, net_mode='resnet', in_channels=4, numclass=10) -> None:
        super(resnet_nonlocal, self).__init__()

        # self.encoder_new =  nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
            
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
            
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
            
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
            
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.AdaptiveAvgPool2d((6, 6))
        # )
        self.encoder_new = EncoderInner(
            net_mode, n_blocks=int(blocks), input_nc=input_nc)
        self.encoder_local = EncoderInner(
            net_mode, n_blocks=int(blocks), input_nc=input_nc)
        # self.encoder_local = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
            
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
            
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
            
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
            
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.AdaptiveAvgPool2d((6, 6))
        # )
        self.block = nn.Sequential(
            nn.Tanh()
        )
        self.Nonlocal_1 = NonLocalBlockND(in_channels)
        self.classfier = Classfier(feature_dim, 256, 128)
        self.out = nn.Sequential(
            nn.Linear(128, numclass)
        )

    def forward(self, x):
        x_local = self.encoder_local(x)
        x_new = self.encoder_new(x)

        x_combine_1 = self.Nonlocal_1(x_new,x_local)

        x_combine_1 = self.block(x_combine_1)

        x_combine_ = torch.flatten(x_combine_1, start_dim=1)
        logs = self.out(self.classfier(x_combine_))

        return logs

    def change_paras(self):
        self.encoder_local.load_state_dict(self.encoder_new.state_dict())


class EncoderInner(nn.Module):
    def __init__(self, net_mode, n_blocks=2, input_nc=3) -> None:
        super().__init__()
        self.net_mode = net_mode
        if net_mode == 'resnet':
            self.encoder = ResnetGenerator(
                input_nc=input_nc, n_blocks=n_blocks)
        elif net_mode == 'alexnet':
           self.encoder  = nn.Sequential(
                OrderedDict([
                    ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                    ('bn1', nn.BatchNorm2d(64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
           
                    ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                    ('bn2', nn.BatchNorm2d(192)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
           
                    ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                    ('bn3', nn.BatchNorm2d(384)),
                    ('relu3', nn.ReLU(inplace=True)),
            
                    ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                    ('bn4', nn.BatchNorm2d(256)),
                    ('relu4', nn.ReLU(inplace=True)),
          
                    ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                    ('bn5', nn.BatchNorm2d(256)),
                    ('relu5', nn.ReLU(inplace=True)),
                    ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2))]))

    def forward(self, x):
        x = self.encoder(x)
        return x


class Classfier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class) -> None:
        super().__init__()
        self.classfier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, num_class)
        )
        # for layer in chain(self.classfier):
        #     if isinstance(layer, (nn.Conv2d, nn.Linear)):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.classfier(x)


class NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            # nn.init.xavier_uniform_(self.W[0].weight)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            # nn.init.xavier_uniform_(self.W.weight)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            # self.theta = nn.Sequential(self.theta, max_pool_layer)

    def forward(self, my_local, my_global):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = my_local.size(0)
        # set g == global
        g_x = self.g(my_global)
        g_x = g_x.view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(my_local).view(
            batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(my_global).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # print(f.shape)
        # 归一化后的attention map
        f_div_C = F.softmax(f, dim=-1)
       # attention * value
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *my_local.size()[2:])
        W_y = self.W(y)
        z = W_y + my_local
        return z


class   ResnetGenerator(nn.Module):

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2, use_bias=True, padding_type='reflect'):
        super().__init__()

        model = [nn.Conv2d(input_nc, 16, kernel_size=3, padding=0, bias=use_bias),
                 norm_layer(16),
                 nn.ReLU(True)]

        model += [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=use_bias),
                  norm_layer(32),
                  nn.ReLU(True)]

        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(32, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.Conv2d(32, 4, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)

        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout=False, use_bias=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class resnet(nn.Module):
    def __init__(self, blocks=2, input_nc=3, feature_dim=784, net_mode='resnet', in_channels=4, numclass=10, KD=False, projection=False) -> None:
        super(resnet, self).__init__()

        self.encoder_new = EncoderInner(
            net_mode, n_blocks=int(blocks), input_nc=input_nc)

        self.block = nn.Sequential(
            nn.Tanh()
        )
        self.classfier = Classfier(feature_dim, 256, 128)
        self.out = nn.Sequential(
            nn.Linear(128, numclass)
        )
        self.projection=projection
        if projection:
            self.p1 = nn.Linear(128, 128)
            self.p2 = nn.Linear(128, 128)

        self.KD = KD
    def forward(self, x):
        x_new = self.encoder_new(x)
        # x_new = self.encoder_new_0(x)
        # x_new = self.encoder_new_2(x_new)
        # x_new = self.encoder_new_3(x_new)
        # x_new = self.encoder_new_4(x_new)


        x_new = self.block(x_new)
        x_combine_ = torch.flatten(x_new, start_dim=1)
        # print(x_combine_.shape)
        # input("wait")
        x_f = self.classfier(x_combine_)
        if self.projection:
            x_f = self.p1(x_f)
            x_f = self.p2(x_f)
        logs = self.out(x_f)
        if self.KD:
            return x_f, logs
        return logs
