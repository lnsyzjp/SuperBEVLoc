import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from tools.utils.load_pcs import *
from tools.utils.read_bin import *
from modules.gemlayer import GeM

def l2_normalize(x,ratio=1.0,axis=1):
    norm=torch.unsqueeze(torch.clamp(torch.norm(x,2,axis),min=1e-6),axis)
    x=x/norm*ratio
    return x

class VanillaLightCNN(nn.Module):
    def __init__(self):
        super(VanillaLightCNN, self).__init__()

        self.conv0=nn.Sequential(
            nn.Conv2d(2,16,5,1,2,bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(32,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            # nn.ReLU(inplace=True),
            #
            # nn.Conv2d(32, 32, 5, 1, 2, bias=False),
            # nn.InstanceNorm2d(32),
        )

    def forward(self, x):
        x=self.conv1(self.conv0(x))
        # print('x shape',x.size())
        x=l2_normalize(x,axis=1)
        # print('x normalize',x.size())
        return x

class BilinearGCNN(nn.Module):
    def __init__(self):
        super(BilinearGCNN, self).__init__()
        self.network1_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network1_embed1_relu = nn.ReLU(True)

        self.network1_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network1_embed2_relu = nn.ReLU(True)

        self.network1_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 8, 3, 1, 1),
        )

        ###########################
        self.network2_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network2_embed1_relu = nn.ReLU(True)

        self.network2_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network2_embed2_relu = nn.ReLU(True)

        self.network2_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1),
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''

        x1 = self.network1_embed1_relu(self.network1_embed1(x) + self.network1_embed1_short(x))
        x1 = self.network1_embed2_relu(self.network1_embed2(x1) + self.network1_embed2_short(x1))
        x1 = self.network1_embed3(x1)

        x2 = self.network2_embed1_relu(self.network2_embed1(x) + self.network2_embed1_short(x))
        x2 = self.network2_embed2_relu(self.network2_embed2(x2) + self.network2_embed2_short(x2))
        x2 = self.network2_embed3(x2)

        # print('x1,x2 shape',x1.size(),x2.size())
        x1 = x1.reshape(x1.size()[0],8, 10000)
        x2 = x2.reshape(x2.size()[0],16, 10000).permute(0,2, 1)  # b*n,25,16
        x = torch.bmm(x1, x2)#.reshape(,-1)  # b*n,8,25

        # print('bmm', x.size())
        x = x.reshape(-1,128)
        # x=l2_normalize(x,axis=2)
        return x

class SuperBEV(nn.Module):
    def __init__(self, height=401, width=401, channels=2):
        super(SuperBEV, self).__init__()
        self.gem = GeM()

        self.VanillaLightCNN = VanillaLightCNN()
        self.BilinearGCNN = BilinearGCNN()

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.VanillaLightCNN(x)
        x = self.BilinearGCNN(x)
        # print(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # print(np.shape(x))
        x = self.gem(x)
        return x

if __name__ == "__main__":
    config_filename = '../config//config.yaml'
    config = yaml.safe_load(open(config_filename))

    file_t = '../data/kitti/kitti00.pickle'
    TRAIN_FILE = '../data/kitti/kitti00_evaluation_database.pickle'
    TEST_FILE = '../data/kitti/kitti00_evaluation_query.pickle'
    TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
    TEST_QUERIES = get_queries_dict(TEST_FILE)
    QUERIES = get_queries_dict(file_t)
    # print(TRAINING_QUERIES[0])
    # print(TEST_QUERIES[0])
    # print(QUERIES[0])

    folder = '/home/zjp/dataset/sequences/00/velodyne/'
    im = load_pcbev_file(folder,TRAINING_QUERIES[0]['file'])

    data = torch.tensor(im)
    print('data',np.shape(data))
    data = data.unsqueeze(dim=0)
    data = data.float()
    print('process',np.shape(data))
    model = SuperBEV()
    res = model(data)

    print(np.shape(res))