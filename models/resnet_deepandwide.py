from torch import nn
import torch


class resnet_fedbalance_experimental(nn.Module):
    def __init__(self, model_local,model_server,KD = False,num_class =10 ) -> None:
        super(resnet_fedbalance_experimental, self).__init__()

        self.model_server = model_server
        self.model_local = model_local
        self.KD = KD
        self.softmax = nn.Softmax(dim=1)
        self.embedding = nn.Embedding(num_embeddings=num_class,embedding_dim=128)
        self.classifier = nn.Linear(256+128,num_class)

    def forward(self, x,distance):

        h_local,_ = self.model_local(x)

        distance_feature = self.embedding(distance)

        h_dis = self.classifier(torch.cat((h_local,distance_feature),dim=1))

        h_new = self.model_server(x)
 
        h_combine = h_dis + h_new
        # h_combine = h_local + h_ne   w

        # probs_local = self.softmax(distance*h_local)
        # probs_new = self.softmax(h_new)

        # probs = torch.log((probs_local+probs_new)/2)

        if self.KD:
            return h_local,h_new,h_combine

        return h_combine

class resnet_fedbalance_server_experimental(nn.Module):
    def __init__(self, model_server) -> None:
        super(resnet_fedbalance_server_experimental, self).__init__()

        self.model_server = model_server

    def forward(self, x):

        h_new = self.model_server(x)

        return h_new