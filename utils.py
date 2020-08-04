import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from textwrap import wrap
import re
import itertools
import matplotlib
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from PyEMD import EMD


def transform_fourier(y):
    fft = np.fft.fft(y, n=180) / len(y)
    fft *= 2
    out = []
    out.append(fft[0].real)
    out += fft[1:-1].real
    out += -fft[1:-1].imag
    out = np.array(out)
    out = out - out.min()
    out = out / out.max()
    return torch.tensor(out, dtype=torch.double)


def get_fourier_coeff(x, y):
    fft = np.fft.fft(y, n=180)/len(y)
    fft *= 2
    out = []
    out.append(fft[0].real)
    out += fft[1:-1].real
    out += -fft[1:-1].imag
    return out

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=False):
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    return fig


class Initial_dataset_loader(Dataset):
    def __init__(self, dataset_folder, transforms=None, full=False, ortho=None, normalize=True,
                 image_size=None):
        padding_minimum = 180
        dataframes = []
        labels = []
        add = False
        for file in files(dataset_folder):
            add = True
            prefix = file.split("_")[0]
            if "T" in prefix:
                labels.append(int(prefix[1]) - 1)
            else:
                add = full and add
                if add:
                    labels.append(4)
            if add:
                dataframes.append(pd.read_csv(os.path.join(dataset_folder, file)))
        tensors = []
        if image_size is not None:
            plotter = PlotToImage(image_size)

        for df in dataframes:
            if image_size is None:
                if ortho is None:
                    data = df.iloc[:, 1:].values[:, 0]
                    if normalize:
                        # print("min: {}".format(np.min(data)))
                        data = data - np.min(data)
                        # print("max: {}".format(np.max(data)))
                        data = data / np.max(data)
                    if len(data) > padding_minimum:
                        start = (len(data)-padding_minimum)//2
                        data = data[start:start+padding_minimum]
                    bckg = np.zeros(padding_minimum)
                    bckg[-len(data):] = data
                    tensors.append(torch.tensor(bckg, dtype=torch.double))
                else:
                    x = np.copy(df.iloc[:, 0:].values[:, 0])
                    x = x - x.mean()
                    y = np.copy(df.iloc[:, 1:].values[:, 0])
                    y = ortho(x, y)
                    if normalize:
                        y = y - np.min(y)
                        y = y / np.max(y)
                    tensors.append(torch.tensor(y, dtype=torch.double))
            else:
                data = df.iloc[:, 1:].values[:, 0]
                data = data - np.min(data)
                data = data / np.max(data)
                tensors.append(plotter(torch.tensor(data)))
                # print(tensors[-1].shape)

        self.whole_set = {
            'data': tensors,
            'id': torch.tensor(labels, dtype=torch.long).view(-1)
        }
        self.transforms = transforms
        self.length = len(self.whole_set['id'])

    def get_class_weights(self):
        ids = self.whole_set["id"].numpy()
        unique, counts = np.unique(ids, return_counts=True)
        counts = 1 - (counts / len(ids)) + (1 / len(unique))
        return torch.tensor(counts)

    def get_dataset(self):
        return self.whole_set

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = None
        data = self.whole_set['data'][idx]
        if self.transforms is not None:
            data = self.transforms(data)
        if 'id' in self.whole_set:
            label = self.whole_set['id'][idx].clone().detach()

        return {
            "image": data,
            "label": label
        }


class ShortenOrElongateTransform:
    def __init__(self, min_length=16, max_length=180, probability=0.5, max_multiplier=2, kind="cubic",
                 window_min=16, window_max=-1):
        self.min_length = min_length
        self.max_length = max_length
        self.max_multiplier = max_multiplier
        self.probability = probability
        self.kind = kind
        if window_max >= max_length or window_max < 0:
            self.window_max = max_length
        else:
            self.window_max = window_max

        self.window_min = window_min

    def __call__(self, x: torch.Tensor):
        np_x = np.trim_zeros(x.numpy())
        elongate_available = False
        shorten_available = False
        window_length = min(len(np_x), random.randint(self.window_min, self.window_max))
        window_start = random.randint(0, len(np_x)-window_length)
        prob_elongate = 0
        prob_shorten = 0
        multiplier = random.randint(2, self.max_multiplier)
        shorten_length = window_length//multiplier
        elongate_length = window_length*multiplier - window_length
        if len(np_x) - shorten_length > self.min_length:
            shorten_available = True
            prob_shorten = self.probability/2
        if elongate_length + len(np_x) < self.max_length:
            elongate_available = True
            if shorten_available:
                prob_elongate = self.probability/2
            else:
                prob_elongate = self.probability
        else:
            if shorten_available:
                prob_shorten = self.probability

        roll = random.random()
        if roll <= prob_shorten and shorten_available:
            rest = random.randint(0, multiplier-1)
            window = np_x[window_start:window_start + window_length]
            return_val = np.array([i for num, i in enumerate(window) if num % multiplier == rest])
            # print(window_start, window_length)
            return_val = np.append(np_x[:window_start], np.append(return_val, np_x[window_start+window_length:]))
        elif roll <= prob_elongate+prob_shorten and elongate_available:
            window = np_x[window_start:window_start + window_length]
            interp_func = interpolate.interp1d(np.arange(0, len(window), 1), window, kind=self.kind)
            xnew = np.arange(0, len(window) - 1, 1 / multiplier)
            return_val = np.array(interp_func(xnew))
            return_val = np.append(np_x[:window_start], np.append(return_val, np_x[window_start+window_length:]))
        else:
            return_val = np_x

        ret_shape = np.zeros(self.max_length)
        ret_shape[-len(return_val):] = return_val
        return_val = torch.tensor(ret_shape)
        return return_val


class PlotToImage:
    def __init__(self, size, interpolation="cubic"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x: torch.Tensor):
        np_x = np.trim_zeros(x.numpy())
        background = np.zeros(self.size)
        interp_func = interpolate.interp1d(np.arange(0, len(np_x), 1), np_x, kind=self.interpolation)
        new_t = np.linspace(0, len(np_x)-1, self.size[0])
        new_x = interp_func(new_t)
        for i in range(len(new_t)):
            if i > 0:
                mx = max(int(np.floor(new_x[i-1]*(self.size[1]-1))), int(np.floor(new_x[i]*(self.size[1]-1))))
                mn = min(int(np.floor(new_x[i-1]*(self.size[1]-1))), int(np.floor(new_x[i]*(self.size[1]-1))))
                background[i, mn:mx] = 1
            else:
                background[i, int(np.floor(new_x[i]*(self.size[1]-1)))] = 1
        return torch.tensor(background).unsqueeze(0)


#class TransformToEmd:
#    def __init__(self, length=180):
#        self.length=length
#
#    def __call__(self, x, y):
#        imf = EMD().emd(y, x)
#        mode = imf[0]
#        bckg = np.zeros(self.length)
#        bckg[:len(mode)] = mode
#        return bckg
#

class resampling_dataset_loader(Dataset):
    def __init__(self, dataset_folder, transforms=None, full=False, normalize=True):
        padding_minimum = 180
        dataframes = []
        labels = []
        add = False
        for file in files(dataset_folder):
            add = True
            prefix = file.split("_")[0]
            if "T" in prefix:
                labels.append(int(prefix[1]) - 1)
            else:
                add = full and add
                if add:
                    labels.append(4)
            if add:
                dataframes.append(pd.read_csv(os.path.join(dataset_folder, file)))
        tensors = []

        for df in dataframes:
            data = df.iloc[:, 1:].values[:, 0]
            interp = interpolate.interp1d(np.arange(0, len(data), 1), data,
                                            kind="cubic")
            new_t = np.linspace(0, len(data)-1, padding_minimum)
            data = interp(new_t)

            if normalize:
                data = data - np.min(data)
                data = data / np.max(data)
            tensors.append(torch.tensor(data, dtype=torch.double))
        self.whole_set = {
            'data': tensors,
            'id': torch.tensor(labels, dtype=torch.long).view(-1)
        }
        self.transforms = transforms
        self.length = len(self.whole_set['id'])

    def get_class_weights(self):
        ids = self.whole_set["id"].numpy()
        unique, counts = np.unique(ids, return_counts=True)
        counts = 1 - (counts / len(ids)) + (1 / len(unique))
        return torch.tensor(counts)

    def get_dataset(self):
        return self.whole_set

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = None
        data = self.whole_set['data'][idx]
        if self.transforms is not None:
            data = self.transforms(data)
        if 'id' in self.whole_set:
            label = self.whole_set['id'][idx].clone().detach()

        return {
            "image": data,
            "label": label
        }
