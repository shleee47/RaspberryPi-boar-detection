import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer,'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class finetunePANNs(nn.Module):
    def __init__(self,PANNs_pretrain,class_num):
        super(finetunePANNs, self).__init__()
        self.PANNs = PANNs_pretrain
        self.add_fc1 = nn.Linear(527,class_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.add_fc1)

    def forward(self, input):
        x=  self.PANNs(input)
        embed = x['embedding']
        clipwise_output = torch.sigmoid(self.add_fc1(embed))
        #output_dict = {'clipwise_output': clipwise_output}
        
        return clipwise_output



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x
