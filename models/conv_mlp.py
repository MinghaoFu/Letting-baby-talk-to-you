import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_mlp(nn.Module):
    def __init__(self, args):
        super(conv_mlp, self).__init__()
        self.args = args
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.fc1 = nn.Linear(4960, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(p=0.5)
        
        self.gender_embedding = nn.Linear(self.args.n_gender_classes, self.args.gender_embed_dim)
        self.age_embedding = nn.Linear(self.args.n_age_classes, self.args.age_embed_dim)

    def forward(self, x, gender, age):
        assert self.args.n_gender_classes == gender.shape[1]
        assert self.args.n_age_classes == age.shape[1]
        assert x.shape == (8, 171, 21), "{}".format(x.shape)
        x = self.conv1(x.unsqueeze(1))
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(8, -1, )
        
        gender_embed = self.gender_embedding(gender)
        age_embed = self.age_embedding(age)
        gender_embed = gender_embed.view(-1, self.args.gender_embed_dim)
        age_embed = age_embed.view(-1, self.args.age_embed_dim)
        
        #import pdb;pdb.set_trace()
        x = torch.cat((x, gender_embed, age_embed), dim=1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x