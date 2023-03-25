#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import scipy
from scipy.io import loadmat
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

num_classes = 4

def get_data(root, signal_size, num_samples):
    data_dictionnary = scipy.io.loadmat(root)
    data = torch.as_tensor(data_dictionnary['Vib_Train_data'])
    data_normal = data[0:int(num_samples/10), ].reshape(1, -1)
    data_inner = data[100:int(num_samples/10)+100, ].reshape(1, -1)
    data_rolling = data[200:int(num_samples/10)+200, ].reshape(1, -1)
    data_out = data[300:int(num_samples/10)+300, ].reshape(1, -1)
    data = torch.cat((data_normal, data_inner, data_rolling, data_out), dim=0)
    data = data.reshape(-1, signal_size)
    data = torch.unsqueeze(data, dim=-1)

    label = []
    for i in range(num_classes):
        j = 0
        while j < (data.shape[0] // num_classes):
            label.append(i)
            j += 1

    data = data.tolist()

    return [data, label]

class Simulation_data(object):
    def __init__(self, normalizetype, signal_size, num_samples):
        self.normalizetype = normalizetype
        self.signal_size = signal_size
        self.num_samples = num_samples
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normalizetype),
                Retype(),
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normalizetype),
                Retype(),
            ])
        }

    def data_split(self):
        # get source train and val
        list_data = get_data(root=r'D:\datasets\chenguo_simulation\simulation_data.mat', signal_size=self.signal_size,
                             num_samples=self.num_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_domain:\n', data_pd)
        train_pd, val_pd = train_test_split(data_pd, test_size=0.2, stratify=data_pd["label"])
        train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
        val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

        return train, val, num_classes




