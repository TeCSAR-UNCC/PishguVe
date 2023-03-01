#network.py
import torch
from torch_geometric.nn import GATConv
from utils.gin_conv2 import GINConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, PReLU
import torch.nn.functional as F

import torch.nn as nn
from torch.nn import init
from types import SimpleNamespace

# Attention modules
################################################################################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'lse']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        if self.pool_types[0]!='lse':
            scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        elif  self.pool_types[0]=='lse':
            scale = F.sigmoid( channel_att_sum ).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'lse'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


################################################################################################################################
    
CBAM_Dropout = 0.25
Linear_Dropout = 0.02

class LinearCBAM(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearCBAM, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.cbam = CBAM(out_features)
        self.dropout1 = nn.Dropout(p=CBAM_Dropout)
        self.dropout2 = nn.Dropout(p=CBAM_Dropout)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.dropout1(x)
        x = self.cbam(x)
        x = x.reshape(x.shape[0], x.shape[1])
        x = self.dropout2(x)
        return x


class GINN3(torch.nn.Module):
    def __init__(self):
        super(GINN3, self).__init__()
        
        input_ch = 120  # input length = 15 => 30*8 = 240
        output_ch = int(input_ch/2)
        
        
        self.l1 = LinearCBAM(input_ch, output_ch)
        self.dropout1 = nn.Dropout(p=Linear_Dropout)

        input_ch = output_ch
        output_ch = int(input_ch/2)
        
        self.l2 = LinearCBAM(input_ch, output_ch)
        

    def forward(self, x):
        
        
        x = F.leaky_relu(self.l1(x))
        
        x = x.reshape(60,-1).t()
        
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.l2(x))
        
        return x
##############################################################################################
##############################################################################################



class GINN(torch.nn.Module):
    def __init__(self):
        super(GINN, self).__init__()
        
        # input_ch = 64 # input length = 8 => 8*8   = 64
        # input_ch = 120  # input length = 15 => 15*8 = 120
        input_ch = 120  # input length = 15 => 30*8 = 240
        output_ch = int(input_ch/2)
        self.l1 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = int(input_ch/2)
        self.l2 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        
        return x

class NetGINConv(torch.nn.Module):
    def __init__(self, num_features, output_size):

        super(NetGINConv, self).__init__()
        self.num_cords = 2
        self.input_steps = int(num_features/self.num_cords)

        
        input_ch = 2
        
        output_ch = 64
        
        ###################################
        
        self.conv2Da = torch.nn.Conv2d(input_ch, output_ch, (2, 2),stride=3)
        torch.nn.init.xavier_uniform_(self.conv2Da.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        
        self.cbam_a = CBAM( output_ch, 16)
        

        input_ch = output_ch
        output_ch = output_ch*2
        
        self.conv2Db = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Db.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam_b = CBAM( output_ch, 16)
        

        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Dc = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
       
        torch.nn.init.xavier_uniform_(self.conv2Dc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam_c = CBAM( output_ch, 16)
        
        ###########################################    
        self.fc = torch.nn.Linear(int(num_features*2),int(num_features*4))

        nn = GINN()
        nn2 = GINN3()
        
        self.conv1 = GINConv(nn, nn2, train_eps=True)
        self.conv11a = GINConv(nn, nn2, train_eps=True)
        self.conv11b = GINConv(nn, nn2, train_eps=True)

        input_ch = output_ch
        output_ch = output_size
        self.conv2Dd = torch.nn.Conv2d(input_ch, output_ch, (1, 1))

    
    def forward(self, x, x_real, edge_index):
        Prelu_flag = 0
        
        
        x1 = F.leaky_relu(self.fc(x_real))
        
        
        x1 = F.leaky_relu(self.conv1(x1, edge_index))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        
        x = F.leaky_relu(self.conv2Da(x))        
        x = self.cbam_a(x)
        
        x = F.leaky_relu(self.conv2Db(x))
        x = self.cbam_b(x)
        
        x = F.leaky_relu(self.conv2Dc(x))
        x = self.cbam_c(x)
        
        # Prediction        
        x = F.leaky_relu(self.conv2Dd(x))

        return x
