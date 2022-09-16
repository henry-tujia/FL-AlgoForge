from models.resnet_fednonlocal import EncoderInner,Classfier
import torch
from torch import nn


class resnet_fedbalance(nn.Module):
    def __init__(self, blocks=2, input_nc=3, feature_dim=784, net_mode='resnet', in_channels=4, numclass=10) -> None:
        super(resnet_fedbalance, self).__init__()

        self.encoder_new = EncoderInner(net_mode, n_blocks=int(blocks), input_nc=input_nc)
        self.encoder_local = EncoderInner(net_mode, n_blocks=int(blocks), input_nc=input_nc)

        self.block = nn.Sequential(
            nn.Tanh()
        )

        self.projection_local = Classfier(feature_dim, 256, numclass)
        self.projection_new = Classfier(feature_dim, 256, numclass)

    def forward(self, x,distance):
        x_local = self.encoder_local(x)
        x_local = self.block(x_local)
        x_local = torch.flatten(x_local, start_dim=1)
        # logits_local = distance*x_local.mm(self.projection_local.weight.transpose(0, 1))
        h_local = self.projection_local(x_local)

        x_new = self.encoder_new(x)
        x_new = self.block(x_new)
        x_new = torch.flatten(x_new, start_dim=1)
        # logits_new = x_new.mm(self.projection_new.weight.transpose(0, 1))
        h_new = self.projection_new(x_new)

        h_combine = distance*h_local + h_new

        return h_combine

class resnet_server(nn.Module):
    def __init__(self, blocks=2, input_nc=3, feature_dim=784, net_mode='resnet', in_channels=4, numclass=10) -> None:
        super(resnet_server, self).__init__()

        self.encoder_new = EncoderInner(net_mode, n_blocks=int(blocks), input_nc=input_nc)
        self.block = nn.Sequential(
            nn.Tanh()
        )

        # self.projection_new = nn.Linear(feature_dim, numclass)
        self.projection_new = Classfier(feature_dim, 256, numclass)

        # self.out = nn.Sequential(
        #     nn.Linear(numclass, numclass,bias=False)
        # )

    def forward(self, x):

        x_new = self.encoder_new(x)
        x_new = self.block(x_new)
        x_new = torch.flatten(x_new, start_dim=1)
        # logits_new = x_new.mm(self.projection_new.weight.transpose(0, 1))
        h_new = self.projection_new(x_new)

        # logits = self.out(h_new)
        # logits = h_new

        return h_new


class resnet_fedbalance_experimental(nn.Module):
    def __init__(self, model_local,model_server) -> None:
        super(resnet_fedbalance_experimental, self).__init__()

        self.model_server = model_server
        self.model_local = model_local

    def forward(self, x,distance):

        h_local = self.model_local(x)

        h_new = self.model_server(x)

        h_combine = distance*h_local + h_new

        return h_combine

class resnet_fedbalance_server_experimental(nn.Module):
    def __init__(self, model_server) -> None:
        super(resnet_fedbalance_server_experimental, self).__init__()

        self.model_server = model_server

    def forward(self, x):

        h_new = self.model_server(x)

        return h_new