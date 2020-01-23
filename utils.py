import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


class Initial_dataset_loader(Dataset):
    def __init__(self, dataset_folder, transforms=None):

        dataframes = []
        labels =[]

        for file in files(dataset_folder):
            dataframes.append(pd.read_csv(os.path.join(dataset_folder, file)))
            labels.append(int(file[1])-1)
        self.whole_set = {
            'data': torch.tensor([df.iloc[:,1:].values[:,0] for df in dataframes],dtype=torch.float),
            'id': torch.tensor(labels,dtype=torch.long).view(-1)
        }
        self.transforms = transforms
        self.length = len(self.whole_set['id'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = None
        data = self.whole_set['data'][idx]
        #if self.transforms:
        #    image = self.transforms(image)
        if 'id' in self.whole_set:
            label = torch.tensor(self.whole_set['id'][idx])

        return {
            "image": data,
            "label": label
        }


