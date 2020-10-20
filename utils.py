import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from textwrap import wrap
import re
from pathlib import Path
import itertools
import matplotlib
import pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from PyEMD import EMD

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def learning_rate_with_decay(lr, batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


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
    fft = np.fft.fft(y, n=180) / len(y)
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
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure() 

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
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
                        start = (len(data) - padding_minimum) // 2
                        data = data[start:start + padding_minimum]
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
        window_start = random.randint(0, len(np_x) - window_length)
        prob_elongate = 0
        prob_shorten = 0
        multiplier = random.randint(2, self.max_multiplier)
        shorten_length = window_length // multiplier
        elongate_length = window_length * multiplier - window_length
        if len(np_x) - shorten_length > self.min_length:
            shorten_available = True
            prob_shorten = self.probability / 2
        if elongate_length + len(np_x) < self.max_length:
            elongate_available = True
            if shorten_available:
                prob_elongate = self.probability / 2
            else:
                prob_elongate = self.probability
        else:
            if shorten_available:
                prob_shorten = self.probability

        roll = random.random()
        if roll <= prob_shorten and shorten_available:
            rest = random.randint(0, multiplier - 1)
            window = np_x[window_start:window_start + window_length]
            return_val = np.array([i for num, i in enumerate(window) if num % multiplier == rest])
            # print(window_start, window_length)
            return_val = np.append(np_x[:window_start], np.append(return_val, np_x[window_start+window_length:]))
        elif roll <= prob_elongate + prob_shorten and elongate_available:
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
    def __init__(self, dataset_folder, transforms=None, full=True, normalize=True,
                    siamese=False, include_artificial_ae = False, artificial_ae_examples=2000,
                    multilabel = False, multilabel_mapping_path = None, multilabel_labels_path = None):
        self.siamese = siamese
        self.multilabel = multilabel
        self.artificial_ae = include_artificial_ae
        self.normalize = normalize
        padding_minimum = 180

        self._create_classes_dict(multilabel_labels_path, multilabel_mapping_path)

        dataframes, self.labels, self.labels_ml = self._get_csv_dataframes_and_labels(dataset_folder,
                                                multilabel_labels_path, full)

        self.tensors, self.tensors_abp = self._get_tensors(dataframes, padding_minimum)

        if artificial_ae_examples is not None:
            self._create_artificial_ae_examples(artificial_ae_examples)
        else:
            self._create_artificial_ae_examples(0)

        self.set_parameters(siamese, multilabel, include_artificial_ae, transforms)
        # print(len(self.tensors))
        # print(len(self.tensors_abp))
        # print(len(self.labels))

    def set_parameters(self, siamese, multilabel, include_artificial_ae, transforms):
        self.siamese = siamese
        self.multilabel = multilabel
        self.artificial_ae = include_artificial_ae

        if siamese:
            if multilabel:
                labels = torch.stack(self.labels_ml)
            else:
                labels = torch.tensor(self.labels, dtype=torch.long).view(-1)
            if include_artificial_ae:
                tensors = self.tensors + self.artificial_icp
                tensors_abp = self.tensors_abp + self.artificial_abp
                if multilabel:
                    labels = torch.cat((labels, self.artificial_lables_ml))
                else:
                    labels = torch.cat((labels, self.artificial_labels))
            else:
                tensors = self.tensors
                tensors_abp = self.tensors_abp
            self.whole_set = {
                'data_icp': tensors,
                'data_abp': tensors_abp,
                'id': labels
            }
        else:
            if multilabel:
                labels = torch.stack(self.labels_ml)
            else:
                labels = torch.tensor(self.labels, dtype=torch.long).view(-1)
            if include_artificial_ae:
                tensors = self.tensors + self.artificial_icp
                if multilabel:
                    labels = torch.cat((labels, self.artificial_lables_ml))
                else:
                    labels = torch.cat((labels, self.artificial_labels))
            else:
                tensors = self.tensors
            self.whole_set = {
                'data_icp': tensors,
                'id': labels
            }
        self.transforms = transforms
        self.length = len(self.whole_set['id'])

    def _create_classes_dict(self, multilabel_labels_path, multilabel_mapping_path):
        classes_dict = [None]
        if multilabel_labels_path is not None:
            ml_labels = pd.read_csv(multilabel_labels_path)

            if multilabel_mapping_path is not None:
                ml_mapping = pd.read_csv(multilabel_mapping_path)
            classes_dict = {}
            for row in ml_labels.iterrows():
                row = row[1]
                pulse_name = row["PULSE"].strip(".csv")+".csv"
                pulse_primary = row["PRIMARY"]
                pulse_secondary = row["SECONDARY"]
                if multilabel_mapping_path is not None:
                    pulse_name = ml_mapping.query("extended_name == \'" + pulse_name + "\'")["siam_name"]
                if not isinstance(pulse_name, str):
                    pulse_name = pulse_name.values[0]
                classes_dict[pulse_name] = (pulse_primary, pulse_secondary)

        self.classes_dict = classes_dict

    def _get_csv_dataframes_and_labels(self, dataset_folder, multilabel_labels_path, full):
        dataframes = []
        labels_ml = []
        labels = []
        add = False
        for file in files(dataset_folder):
            add = True
            prefix = file.split("_")[0]
            if file in self.classes_dict:
                prim, sec = self.classes_dict[file]
                out = torch.zeros(5, dtype = torch.float)
                out[prim-1] = 1
                out[sec-1] = 1
                labels_ml.append(out)
            else:
                if "T" in prefix:
                    out = torch.zeros(5, dtype=torch.float)
                    out[int(prefix[1]) - 1] = 1
                    labels_ml.append(out)
                else:
                    labels_ml.append(torch.tensor([0, 0, 0, 0, 1], dtype=torch.float))

            if multilabel_labels_path is not None:
                if file in self.classes_dict:
                    prim, sec = self.classes_dict[file]
                    labels.append(prim-1)
                    if prim - 1 == 4:
                        add = full and add
                elif "T" in prefix:
                    labels.append(int(prefix[1]) - 1)
                else:
                    add = full and add
                    if add:
                        labels.append(4)
            elif "T" in prefix:
                labels.append(int(prefix[1]) - 1)
            else:
                add = full and add
                if add:
                    labels.append(4)
            if add:
                dataframes.append(pd.read_csv(os.path.join(dataset_folder, file)))
        return dataframes, labels, labels_ml

    def _get_tensors(self, dataframes, padding_minimum):
        tensors = []
        tensors_abp = []

        for df in dataframes:
            data = df.iloc[:, 1:].values[:, 0]

            interp = interpolate.interp1d(np.arange(0, len(data), 1), data,
                                            kind="cubic")
            new_t = np.linspace(0, len(data)-1, padding_minimum)
            data = interp(new_t)

            if self.normalize:
                data = data - np.min(data)
                data = data / np.max(data)

            if self.siamese:
                data_abp = df.iloc[:, 1:].values[:, 1]
                interp_abp = interpolate.interp1d(np.arange(0, len(data_abp), 1), data_abp,
                                                kind="cubic")
                data_abp = interp_abp(new_t)
                if self.normalize:
                    data_abp = data_abp - np.min(data_abp)
                    if np.max(data_abp) != 0:
                        data_abp = data_abp / np.max(data_abp)

                tensors_abp.append(torch.tensor(data_abp, dtype=torch.float))

            tensors.append(torch.tensor(data, dtype=torch.float))
        return tensors, tensors_abp

    def get_class_weights(self):
        ids = self.whole_set["id"].numpy()
        unique, counts = np.unique(ids, return_counts=True)
        counts = 1 - (counts / len(ids)) + (1 / len(unique))
        return torch.tensor(counts)

    def get_dataset(self):
        return self.whole_set

    def _get_noise(self, max_amp=0.5, min_sines = 1, max_sines = 8, max_hz = 10, length=180):
        noise = np.zeros(length)
        no_of_sines = random.randint(min_sines, max_sines)
        for i in range(no_of_sines):
            x = np.linspace(-np.pi, np.pi, length)
            f = random.randint(1, max_hz)
            fi = random.random()
            noise += np.sin(f * x + fi)

        noise = noise-min(noise)
        noise = noise/max(noise)
        noise = max_amp * noise + 1
        return noise

    def _apply_noise(self, signal, signal_abp=None):
        noise = self._get_noise(length=len(signal))
        if signal_abp is not None:
            return (signal*noise)/max(signal*noise), (signal_abp*noise)/max(signal_abp*noise)
        return (signal*noise)/max(signal*noise)

    def _create_artificial_ae_examples(self, no_of_examples = 2000):
        if no_of_examples <= 0:
            return
        df = pd.DataFrame({
            "id": self.labels,
            "data_icp": self.tensors,
            "data_abp": self.tensors_abp
        })
        queried = df.query("id != 4")
        cho = random.choices(np.arange(len(queried)), k=no_of_examples)
        self.artificial_labels = []
        self.artificial_lables_ml = []
        self.artificial_icp = []
        self.artificial_abp = []
        for choice in cho:
            row = queried.iloc[choice]
            ID = row["id"]
            if self.siamese:
                icp_new, abp_new = self._apply_noise(row["data_icp"], row["data_abp"])
                self.artificial_icp.append(torch.tensor(icp_new, dtype=torch.float))
                self.artificial_abp.append(torch.tensor(abp_new, dtype=torch.float))
            else:
                icp_new = self._apply_noise(row["data_icp"])
                self.artificial_icp.append(torch.tensor(icp_new, dtype=torch.float))

            lbl = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)
            lbl[np.argmax(ID)] = 1
            self.artificial_lables_ml.append(lbl)
            self.artificial_labels.append(4)

        self.artificial_lables_ml = torch.stack(self.artificial_lables_ml)
        self.artificial_labels = torch.tensor(self.artificial_labels, dtype=torch.long).view(-1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = None
        data_icp = self.whole_set['data_icp'][idx]
        if self.siamese:
            data_abp = self.whole_set['data_abp'][idx]
        if self.transforms is not None:
            if self.siamese:
                data_icp, data_abp = self.transforms((data_icp, data_abp))
            else:
                data_icp = self.transforms(data_icp)
        if 'id' in self.whole_set:
            label = self.whole_set['id'][idx].clone().detach()

        if self.siamese:
            return {
                "data_icp": data_icp,
                "data_abp": data_abp,
                "label": label
            }
        else:
            return {
                "data_icp": data_icp,
                "label": label
            }

class Memory_efficient_loader(Dataset):
    def __init__(self, dataset_folder, normalize=True, siamese=True, csv=False):
        self.siamese = siamese
        self.normalize = normalize
        self.padding_minimum = 180
        data_folder = Path(str(dataset_folder))
        if csv:
            data_files = list(data_folder.glob('*.csv'))
        else:
            data_files = list(data_folder.glob('*.pkl'))
        self.length = len(data_files)
        print(self.length)
        self.data_files = data_files
        self.csv = csv

    def _load_file(self, idx):
        file_path = self.data_files[idx]
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values[:, 0]

        interp = interpolate.interp1d(np.arange(0, len(data), 1), data,
                                        kind="cubic")
        new_t = np.linspace(0, len(data)-1, self.padding_minimum)
        data = interp(new_t)

        if self.normalize:
            data = data - np.min(data)
            if np.max(data) != 0:
                data = data / np.max(data)

        icp = torch.tensor(data, dtype=torch.float)
        if self.siamese:
            data_abp = df.iloc[:, 1:].values[:, 1]
            interp_abp = interpolate.interp1d(np.arange(0, len(data_abp), 1), data_abp,
                                            kind="cubic")
            data_abp = interp_abp(new_t)
            if self.normalize:
                data_abp = data_abp - np.min(data_abp)
                if np.max(data_abp) != 0:
                    data_abp = data_abp / np.max(data_abp)

            abp = torch.tensor(data_abp, dtype=torch.float)
            return icp, abp
        return icp, None

    def load_pickle(self, idx):
        file_path = self.data_files[idx]
        with open(file_path, "rb") as file:
            loaded = pickle.load(file)
        if self.siamese:
            return loaded["icp"], loaded["abp"]
        else:
            return loaded["icp"], None

    def __len__(self):
        return self.length

    def set_parameters(self, siamese):
        self.siamese = siamese

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif not isinstance(idx, list):
            idx = [idx]
        icps = []
        abps = []
        for id in idx:
            if self.csv:
                icp, abp = self._load_file(id)
                icps.append(icp)
                abps.append(abp)
            else:
                icp, abp = self.load_pickle(id)
                icps += icp
                abps += abp
        if self.csv:
            icps = torch.cat(icps)
            abps = torch.cat(abps)
        else:
            icps = torch.tensor(icps, dtype=torch.float)
            abps = torch.tensor(abps, dtype=torch.float)
        if self.siamese:
            return {
                "data_icp": icps,
                "data_abp": abps
                }
        else:
            return {
                "data_icp": icps
                }
