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
    def __init__(self, dataset_folder, transforms=None, full=False):
        padding_minimum = torch.zeros(180)
        dataframes = []
        labels =[]
        add = False
        for file in files(dataset_folder):
            add = True
            prefix = file.split("_")[0]
            if "T" in prefix:
                labels.append(int(prefix[1]) - 1)
            elif "A" in prefix:
                add = full and add
                if add:
                    labels.append(4)
            else:
                add = full and add
                if add:
                    labels.append(5)
            if add:
                dataframes.append(pd.read_csv(os.path.join(dataset_folder, file)))
        tensors = []
        for df in dataframes:
            tensors.append(torch.tensor(df.iloc[:,1:].values[:,0], dtype=torch.double))
        tensors.append(padding_minimum)
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        tensors = tensors[:-1]
        print(tensors.shape)
        self.whole_set = {
            'data': tensors,
            'id': torch.tensor(labels, dtype=torch.long).view(-1)
        }
        self.transforms = transforms
        self.length = len(self.whole_set['id'])

    def get_class_weights(self):
        ids = self.whole_set["id"].numpy()
        unique, counts = np.unique(ids, return_counts=True)
        counts = 1 - (counts/len(ids))+(1/len(unique))
        return torch.tensor(counts)

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
            label = self.whole_set['id'][idx].clone().detach()

        return {
            "image": data,
            "label": label
        }


