from torch import nn


class resnet_fedbalance_experimental(nn.Module):
    def __init__(self, model_local,model_global,KD = False) -> None:
        super(resnet_fedbalance_experimental, self).__init__()

        self.model_global = model_global
        self.model_local = model_local
        self.KD = KD
        self.softmax = nn.Softmax(dim=1)
        self.confidence = nn.Linear(10,10)

    def forward(self, x,distance):

        h_local = self.model_local(x)

        h_new = self.model_global(x)

        # print(torch.kl_div(a.log(),b).mean())
 
        h_combine = distance*h_local + h_new
        # h_combine = h_local + h_ne   w

        # probs_local = self.softmax(distance*h_local)
        # probs_new = self.softmax(h_new)

        # probs = torch.log((probs_local+probs_new)/2)

        if self.KD:
            return h_local,h_new,h_combine

        return h_combine

class resnet_fedbalance_server_experimental(nn.Module):
    def __init__(self, model_global) -> None:
        super(resnet_fedbalance_server_experimental, self).__init__()

        self.model_global = model_global

    def forward(self, x):

        h_new = self.model_global(x)
 
        return h_new