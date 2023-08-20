from torch import nn


class resnet_fedbalance_experimental(nn.Module):
    def __init__(self, model_local, model_global, KD=False) -> None:
        super(resnet_fedbalance_experimental, self).__init__()

        self.model_global = model_global
        self.model_local = model_local
        self.KD = KD

    def forward(self, x, distance):

        h_local = self.model_local(x)

        h_new = self.model_global(x)

        h_combine = distance*h_local + h_new

        return h_combine


class resnet_triple(nn.Module):
    def __init__(self, model_private, model_local, model_global) -> None:
        super(resnet_triple, self).__init__()

        self.model_global = model_global
        self.model_local = model_local
        self.model_private = model_private

    def forward(self, x, distance):

        h_local = self.model_local(x)

        h_new = self.model_global(x)

        h_combine = distance*h_local + h_new

        return h_combine


class resnet_fedbalance_server_experimental(nn.Module):
    def __init__(self, model_global) -> None:
        super(resnet_fedbalance_server_experimental, self).__init__()

        self.model_global = model_global

    def forward(self, x):

        h_new = self.model_global(x)

        return h_new
