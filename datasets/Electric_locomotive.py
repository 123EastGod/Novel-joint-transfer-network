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
    # data
    data_dictionnary = scipy.io.loadmat(root)
    normal = torch.as_tensor(data_dictionnary['C1'][0:signal_size*num_samples, ]).reshape(1, -1)
    inner_race = torch.as_tensor(data_dictionnary['C4'][0:signal_size*num_samples, ]).reshape(1, -1)
    roller_defect = torch.as_tensor(data_dictionnary['C5'][0:signal_size*num_samples, ]).reshape(1, -1)
    serious_outer_race = torch.as_tensor(data_dictionnary['C3'][0:signal_size*num_samples, ]).reshape(1, -1)

    data = torch.cat((normal, inner_race, roller_defect, serious_outer_race), dim=0)
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


class Electric_locomotive_data(object):
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
        list_data = get_data(root=r'D:\datasets\株洲机车轴承\株洲有用的九组data\zhuzhou_9.mat', signal_size=self.signal_size,
                             num_samples=self.num_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        train_pd, val_pd = train_test_split(data_pd, test_size=0.2, stratify=data_pd["label"])
        train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
        val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

        return train, val


