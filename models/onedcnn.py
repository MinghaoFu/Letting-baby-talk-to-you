import torch
import torch.nn as nn
import torch.nn.functional as F


def call_bn(bn, x):
    return bn(x)


class onedcnn(nn.Module):
    def __init__(self, n_classes=5, dropout=0.5):
        super(onedcnn, self).__init__()
        self.dropout_rate= dropout
        self.c1=nn.Conv2d(1,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_classes)

        self.fc1= nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(512)
        self.bn7=nn.BatchNorm2d(256)
        self.bn8=nn.BatchNorm2d(128)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x):
        # print(x.shape)
        x = torch.unsqueeze(x, 1)
        # print(x.shape)
        h=x
        # print(x.shape)
        h=self.c1(h)
        # print(x.shape)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        # print(h.shape)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        # print(h.shape)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)
        # print(h.shape)
        h=self.c7(h)
        # print(h.shape)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        # print("c8",h.shape)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        # print("c8",h.shape)
        # h=self.c9(h)
        # # print("c9", h.shape)
        # h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=(h.data.shape[2],1))
        
        # print(h.shape)
        h = h.view(h.size(0), h.size(1))
        # print(h.shape)
        logit=self.l_c1(h)

        return logit



    def forward_contrastive(self, x):
        # print(x.shape)
        x = torch.unsqueeze(x, 1)
        # print(x.shape)
        h=x
        # print(x.shape)
        h=self.c1(h)
        # print(x.shape)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        # print(h.shape)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        # print(h.shape)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)
        # print(h.shape)
        h=self.c7(h)
        # print(h.shape)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        # print("c8",h.shape)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        # print("c8",h.shape)
        # h=self.c9(h)
        # # print("c9", h.shape)
        # h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=(h.data.shape[2],1))
        
        # print(h.shape)
        h = h.view(h.size(0), h.size(1))
        h=self.fc1(h)
        h=F.leaky_relu(h, negative_slope=0.01)
        h=self.fc2(h)
        h = F.normalize(h, dim=1)

        return h
   
