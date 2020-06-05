import torch.nn as nn
import torch

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def l2(input,axis=1):
    norm = torch.norm(input, 2, axis,True)
    output = torch.div(input, norm)
    return output

class FConv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, out_padding=0):
        super(FConv_block, self).__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False, dilation=1, output_padding=out_padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Reshape(nn.Module): 
    def __init__(self, shape): 
        super(Reshape, self).__init__() 
        self.shape = shape
    
    def forward(self, x): 
        return x.view(self.shape)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__() 
        self.enc = nn.Sequential(
            Conv_block(3, 32, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(32, 64, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(64, 64, (3, 3), (2, 2), (1, 1), 1),
            Conv_block(64, 64, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(64, 128, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(128, 128, (3, 3), (2, 2), (1, 1), 1),
            Conv_block(128, 96, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(96, 192, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(192, 192, (3, 3), (2, 2), (0, 0), 1),
            Conv_block(192, 128, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(128, 256, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(256, 256, (3, 3), (2, 2), (1, 1), 1),
            Conv_block(256, 160, (3, 3), (1, 1), (1, 1), 1),
            nn.Conv2d(160, out_channels=320, kernel_size=(3, 3), groups=1, stride=(1, 1), padding=(1, 1), bias=False),
            nn.AvgPool2d(6)            
        )
    def forward(self, x):
        g = self.enc(x)
        g = torch.flatten(g, start_dim=1)
        return g

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__() 
        self.dec = nn.Sequential(
            nn.Linear(320, 6*6*320),
            Reshape((-1, 320, 6, 6)),
            FConv_block(320, 160, (3, 3), 1, 1, 1),
            FConv_block(160, 256, (3, 3), 1, 1, 1),
            FConv_block(256, 256, (3, 3), 2, 1, 1, 1),
            FConv_block(256, 128, (3, 3), 1, 1, 1),
            FConv_block(128, 192, (3, 3), 1, 1, 1),
            FConv_block(192, 192, (3, 3), 2, 1, 1, 1),
            FConv_block(192, 96, (3, 3), 1, 1, 1),
            FConv_block(96, 128, (3, 3), 1, 1, 1),
            FConv_block(128, 128, (3, 3), 2, 1, 1, 1),
            FConv_block(128, 64, (3, 3), 1, 1, 1),
            FConv_block(64, 64, (3, 3), 1, 0, 1),
            FConv_block(64, 64, (3, 3), 2, 1, 1, 1),
            FConv_block(64, 32, (3, 3), 1, 1, 1),
            FConv_block(32, 3, (3, 3), 1, 1, 1),
        )
    def forward(self, g):
        x_dec = self.dec(g)
        return x_dec

class Distillation_R(nn.Module):
    def __init__(self):
        super(Distillation_R, self).__init__() 
        self.R = nn.Sequential(
            nn.Linear(320, 6*6*320),
            Reshape((-1, 320, 6, 6)),
            FConv_block(320, 160, (3, 3), (1, 1), (1, 1), 1),
            FConv_block(160, 256, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(256, 160, (3, 3), (1, 1), (1, 1), 1),
            nn.Conv2d(160, out_channels=320, kernel_size=(3, 3), groups=1, stride=(1, 1), padding=(1, 1), bias=False),
            nn.AvgPool2d(6) 
        )
    def forward(self, g):
        f = self.R(g)
        f = torch.flatten(f, start_dim=1)
        return l2(f)

class FC_softmax(nn.Module):
    def __init__(self, embedding_size=320, classnum=51332):
        super(FC_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings):
        output = torch.mm(embbedings, self.kernel)
        return output

class FTL_model(nn.Module):
    def __init__(self):
        super(FTL_model, self).__init__()
        self.enc = nn.Sequential(
            Conv_block(3, 32, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(32, 64, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(64, 64, (3, 3), (2, 2), (1, 1), 1),
            Conv_block(64, 64, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(64, 128, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(128, 128, (3, 3), (2, 2), (1, 1), 1),
            Conv_block(128, 96, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(96, 192, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(192, 192, (3, 3), (2, 2), (0, 0), 1),
            Conv_block(192, 128, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(128, 256, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(256, 256, (3, 3), (2, 2), (1, 1), 1),
            Conv_block(256, 160, (3, 3), (1, 1), (1, 1), 1),
            nn.Conv2d(160, out_channels=320, kernel_size=(3, 3), groups=1, stride=(1, 1), padding=(1, 1), bias=False),
            nn.AvgPool2d(6)            
        )
        self.dec = nn.Sequential(
            nn.Linear(320, 6*6*320),
            Reshape((-1, 320, 6, 6)),
            FConv_block(320, 160, (3, 3), 1, 1, 1),
            FConv_block(160, 256, (3, 3), 1, 1, 1),
            FConv_block(256, 256, (3, 3), 2, 1, 1, 1),
            FConv_block(256, 128, (3, 3), 1, 1, 1),
            FConv_block(128, 192, (3, 3), 1, 1, 1),
            FConv_block(192, 192, (3, 3), 2, 1, 1, 1),
            FConv_block(192, 96, (3, 3), 1, 1, 1),
            FConv_block(96, 128, (3, 3), 1, 1, 1),
            FConv_block(128, 128, (3, 3), 2, 1, 1, 1),
            FConv_block(128, 64, (3, 3), 1, 1, 1),
            FConv_block(64, 64, (3, 3), 1, 0, 1),
            FConv_block(64, 64, (3, 3), 2, 1, 1, 1),
            FConv_block(64, 32, (3, 3), 1, 1, 1),
            FConv_block(32, 3, (3, 3), 1, 1, 1),
        )
        self.R = nn.Sequential(
            nn.Linear(320, 6*6*320),
            Reshape((-1, 320, 6, 6)),
            FConv_block(320, 160, (3, 3), (1, 1), (1, 1), 1),
            FConv_block(160, 256, (3, 3), (1, 1), (1, 1), 1),
            Conv_block(256, 160, (3, 3), (1, 1), (1, 1), 1),
            nn.Conv2d(160, out_channels=320, kernel_size=(3, 3), groups=1, stride=(1, 1), padding=(1, 1), bias=False),
            nn.AvgPool2d(6) 
        )

    def forward(self, x):
        g = self.enc(x)
        g = torch.flatten(g, start_dim=1)
        x_dec = self.dec(g)
        f = self.R(g)
        f = torch.flatten(f, start_dim=1) 
        return  x_dec, l2(f)

