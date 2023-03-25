#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self,  in_channel=1, ):
        super(CNN, self).__init__()

        # 采样频率较高的数据第一层的卷积核的核大小一定要大一点
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=25),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x


# convnet without the last layer
class cnn_features(nn.Module):
    def __init__(self):
        super(cnn_features, self).__init__()
        self.model_cnn = CNN()
        self.__in_features = 256

    def forward(self, x):
        x = self.model_cnn(x)
        return x

    def outputdim(self):
        return self.__in_features