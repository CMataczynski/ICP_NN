import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import torchvision
import pandas as pd
from sklearn.metrics import f1_score
import time
import numpy as np
import datetime as dt
from torch.utils.data import Dataset
from models.SiameseModels import SiameseNeuralODE, SiameseResNet, SiameseShallowCNN
from utils import plot_confusion_matrix
from torchvision.transforms import Compose
from tqdm import tqdm
from scipy import interpolate
import random

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

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


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(lr, batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy_and_f1(model, dataset_loader,no_classes , device="cpu"):
    total_correct = 0
    labs = []
    preds = []
    for data in dataset_loader:
        x = data['data_icp'].to(device)
        x_abp = data['data_abp'].to(device)
        y = one_hot(np.array(data['label'].numpy()), no_classes)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x, x_abp).cpu().detach().numpy(), axis=1)
        labs += data["label"].tolist()
        preds += predicted_class.tolist()
        total_correct += np.sum(predicted_class == target_class)
    f1 = f1_score(labs, preds, average='weighted')
    return total_correct / len(dataset_loader.dataset), f1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def create_experiment_name(experiment_name, type):
    if type == 0:
        name = "SiameseODE"
    elif type == 1:
        name = "SiameseResnet"
    else:
        name = "SiameseCNN"
    name = experiment_name + "_" + str(dt.date.today()) + "_" + name

    max = 0
    for subdir in os.listdir(os.path.join(os.getcwd(), "experiments")):
        if name in subdir:
            var = int(subdir.split('_')[-1])
            if var > max:
                max = var
    return name + "_" + str(max+1)

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
        tensors_abp = []
        if image_size is not None:
            plotter = PlotToImage(image_size)

        for df in dataframes:
            if image_size is None:
                if ortho is None:
                    data_icp = df.iloc[:, 1:].values[:, 0]
                    data_abp = df.iloc[:, 1:].values[:, 1]

                    if normalize:
                        # print("min: {}".format(np.min(data)))
                        data_icp = data_icp - np.min(data_icp)
                        data_abp = data_abp - np.min(data_abp)
                        # print("max: {}".format(np.max(data_icp)))
                        data_icp = data_icp / np.max(data_icp)
                        if max(data_abp) != 0:
                            data_abp = data_abp / np.max(data_abp)
                    if len(data_icp) > padding_minimum:
                        start = (len(data_icp)-padding_minimum)//2
                        data_icp = data_icp[start:start+padding_minimum]
                    if len(data_abp) > padding_minimum:
                        start = (len(data_abp)-padding_minimum)//2
                        data_abp = data_abp[start:start+padding_minimum]
                    bckg = np.zeros(padding_minimum)
                    bckg[-len(data_icp):] = data_icp
                    tensors.append(torch.tensor(bckg, dtype=torch.float))
                    bckg = np.zeros(padding_minimum)
                    bckg[-len(data_abp):] = data_abp
                    tensors_abp.append(torch.tensor(bckg, dtype=torch.float))
                else:
                    x = np.copy(df.iloc[:, 0:].values[:, 0])
                    x = x - x.mean()
                    y = np.copy(df.iloc[:, 1:].values[:, 0])
                    y = ortho(x, y)
                    if normalize:
                        y = y - np.min(y)
                        y = y / np.max(y)
                    tensors.append(torch.tensor(y, dtype=torch.float))
            else:
                data_icp = df.iloc[:, 1:].values[:, 0]
                data_icp = data_icp - np.min(data_icp)
                data_icp = data_icp / np.max(data_icp)
                tensors.append(plotter(torch.tensor(data_icp)))
                data_abp = df.iloc[:, 1:].values[:, 1]
                data_abp = data_abp - np.min(data_abp)
                data_abp = data_abp / np.max(data_abp)
                tensors_abp.append(plotter(torch.tensor(data_abp)))
                # print(tensors[-1].shape)

        self.whole_set = {
            'data_icp': tensors,
            'data_abp': tensors_abp,
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
        data_icp = self.whole_set['data_icp'][idx]
        data_abp = self.whole_set['data_abp'][idx]
        if self.transforms is not None:
            data_icp, data_abp = self.transforms((data_icp, data_abp))
        if 'id' in self.whole_set:
            label = self.whole_set['id'][idx].clone().detach()

        return {
            "data_icp": data_icp,
            "data_abp": data_abp,
            "label": label
        }


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
        tensors_abp = []
        for df in dataframes:
            data_icp = df.iloc[:, 1:].values[:, 0]
            data_abp = df.iloc[:, 1:].values[:, 1]
            interp_icp = interpolate.interp1d(np.arange(0, len(data_icp), 1), data_icp,
                                            kind="cubic")
            interp_abp = interpolate.interp1d(np.arange(0, len(data_abp), 1), data_abp,
                                            kind="cubic")

            new_t = np.linspace(0, len(data_icp)-1, padding_minimum)
            data_icp = interp_icp(new_t)
            data_abp = interp_abp(new_t)

            if normalize:
                data_icp = data_icp - np.min(data_icp)
                data_icp = data_icp / np.max(data_icp)
                data_abp = data_abp - np.min(data_abp)
                if np.max(data_abp) != 0:
                    data_abp = data_abp / np.max(data_abp)
            tensors.append(torch.tensor(data_icp, dtype=torch.float))
            tensors_abp.append(torch.tensor(data_abp, dtype=torch.float))
        self.whole_set = {
            'data_icp': tensors,
            'data_abp': tensors_abp,
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
        data_icp = self.whole_set['data_icp'][idx]
        data_abp = self.whole_set['data_abp'][idx]
        if self.transforms is not None:
            data_icp, data_abp = self.transforms((data_icp, data_abp))
        if 'id' in self.whole_set:
            label = self.whole_set['id'][idx].clone().detach()

        return {
            "data_icp": data_icp,
            "data_abp": data_abp,
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

    def __call__(self, xinput):
        x, x1 = xinput
        np_x = np.trim_zeros(x.numpy())
        np_x1 = np.trim_zeros(x1.numpy())
        length = max(len(np_x), len(np_x1))
        elongate_available = False
        shorten_available = False
        window_length = min(length, random.randint(self.window_min, self.window_max))
        if window_length < self.window_min:
            return x, x1
        window_start = random.randint(0, length-window_length)
        prob_elongate = 0
        prob_shorten = 0
        multiplier = random.randint(2, self.max_multiplier)
        shorten_length = window_length//multiplier
        elongate_length = window_length*multiplier - window_length
        if length - shorten_length > self.min_length:
            shorten_available = True
            prob_shorten = self.probability/2
        if elongate_length + length < self.max_length:
            elongate_available = True
            if shorten_available:
                prob_elongate = self.probability/2
            else:
                prob_elongate = self.probability
        else:
            if shorten_available:
                prob_shorten = self.probability

        roll = random.random()
        try:
            if roll <= prob_shorten and shorten_available:
                rest = random.randint(0, multiplier-1)
                window = np_x[window_start:window_start + window_length]
                return_val = np.array([i for num, i in enumerate(window) if num % multiplier == rest])

                window1 = np_x1[window_start:window_start + window_length]
                return_val1 = np.array([i for num, i in enumerate(window1) if num % multiplier == rest])

                return_val = np.append(np_x[:window_start], np.append(return_val, np_x[window_start+window_length:]))
                return_val1 = np.append(np_x1[:window_start], np.append(return_val1, np_x1[window_start+window_length:]))
            elif roll <= prob_elongate+prob_shorten and elongate_available:
                window = np_x[window_start:window_start + window_length]
                interp_func = interpolate.interp1d(np.arange(0, len(window), 1), window, kind=self.kind)
                xnew = np.arange(0, len(window) - 1, 1 / multiplier)
                return_val = np.array(interp_func(xnew))
                return_val = np.append(np_x[:window_start], np.append(return_val, np_x[window_start+window_length:]))
                window1 = np_x1[window_start:window_start + window_length]
                interp_func1 = interpolate.interp1d(np.arange(0, len(window1), 1), window1, kind=self.kind)
                xnew1 = np.arange(0, len(window1) - 1, 1 / multiplier)
                return_val1 = np.array(interp_func1(xnew1))
                return_val1 = np.append(np_x1[:window_start], np.append(return_val1, np_x1[window_start+window_length:]))
            else:
                return_val = np_x
                return_val1 = np_x1
        except:
            return_val = np_x
            return_val1 = np_x1

        ret_shape = np.zeros(self.max_length)
        ret_shape[-len(return_val):] = return_val
        return_val = torch.tensor(ret_shape, dtype=torch.float)
        ret_shape1 = np.zeros(self.max_length)
        ret_shape1[-len(return_val1):] = return_val1
        return_val1 = torch.tensor(ret_shape1, dtype=torch.float)
        return (return_val, return_val1)


def trainODE(experiment_name, dataset_loaders
            ,class_dict, training_length=150,
            type=0, batch_size=256, lr=0.1):
    '''
    experiment_name: string with name of the experiments
    dataset_loaders: (training loader, testing loader) with type of DataLoader

    '''
    no_classes = len(class_dict.keys())
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    name = create_experiment_name(experiment_name, type)
    writer = SummaryWriter(log_dir='experiments/' + str(name))
    makedirs(os.path.join(os.getcwd(), "experiments", name))

    if type == 0:
        model = SiameseNeuralODE(no_classes).to(device)
    elif type == 1:
        model = SiameseResNet(no_classes).to(device)
    else:
        model = SiameseShallowCNN(no_classes).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    train_loader, test_loader = dataset_loaders
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(lr,
         batch_size, batch_denom=batch_size, batches_per_epoch=batches_per_epoch,
         boundary_epochs=[40, 60, 80],decay_rates=[1, 0.1, 0.01, 0.001]
         )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_acc = 0
    best_f1 = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    running_loss = 0.0
    for itr in tqdm(range(training_length * batches_per_epoch)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        model.train()
        optimizer.zero_grad()
        dct = data_gen.__next__()
        # print(dct)
        x = dct["data_icp"]
        x_abp = dct["data_abp"]
        y = dct["label"]
        x = x.float().to(device)
        x_abp = x_abp.float().to(device)
        y = y.to(device)
        # x = x.unsqueeze(1)
        logits = model(x, x_abp)
        loss = criterion(logits, y)

        if type == 0:
            nfe_forward = model.feature_layers[0].nfe
            model.feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if itr % 10 == 9:
            writer.add_scalar("Loss/train", running_loss / 10, itr)
            running_loss = 0.0
        if type == 0:
            nfe_backward = model.feature_layers[0].nfe
            model.feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if type == 0:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                model.eval()
                val_acc, f1 = accuracy_and_f1(model, test_loader, no_classes, device=device)
                if f1 > best_f1:
                    torch.save({'state_dict': model.state_dict()},
                                os.path.join(os.getcwd(),"experiments", name,
                                            'model_1.pth'))
                    best_f1 = f1
                    best_acc = val_acc
                writer.add_scalar("Accuracy/test", val_acc, itr//batches_per_epoch)
                writer.add_scalar("F1_score/test", f1, itr//batches_per_epoch)
                if type == 0:
                    writer.add_scalar("NFE-F", f_nfe_meter.val, itr//batches_per_epoch)
                    writer.add_scalar("NFE-B", b_nfe_meter.val, itr//batches_per_epoch)

    labs = []
    preds = []
    for data in test_loader:
        x = data['data_icp'].to(device)
        x_abp = data['data_abp'].to(device)
        # x = x.unsqueeze(1)
        y = data["label"].tolist()
        labs += y
        outputs = model(x, x_abp)
        predicted = torch.max(outputs, 1).indices
        preds += predicted.tolist()

    labs = [class_dict[a] for a in labs]
    preds = [class_dict[a] for a in preds]
    writer.add_figure(name + " - Confusion Matrix",
                      plot_confusion_matrix(labs, preds,
                      [class_dict[key] for key in class_dict.keys()]))
    writer.close()
    return [best_acc, best_f1]

if __name__ == "__main__":
    batch_size = 128
    datasets = os.path.join(os.getcwd(), "datasets", "full_siamese_dataset")
    train_dataset_path = os.path.join(datasets, "train")
    test_dataset_path = os.path.join(datasets, "test")

    df = pd.DataFrame({
        "name": [],
        "best_acc": [],
        "best_f1": []
    })
    train_4cls = Initial_dataset_loader(train_dataset_path, full=False,
                                        transforms= Compose([
                                            ShortenOrElongateTransform(min_length=32,
                                                                       max_length=180,
                                                                       probability=0.7,
                                                                       max_multiplier=3)
                                        ]),
                                        normalize=True)
    test_4cls = Initial_dataset_loader(test_dataset_path, full=False, normalize=True)

    res_train_4cls = resampling_dataset_loader(train_dataset_path, full=False,
                                        normalize=True)
    res_test_4cls = resampling_dataset_loader(test_dataset_path, full=False, normalize=True)
    print("Loaded 4 cls")
    train_5cls = Initial_dataset_loader(train_dataset_path, full=True,
                                        transforms= Compose([
                                            ShortenOrElongateTransform(min_length=32,
                                                                       max_length=180,
                                                                       probability=0.7,
                                                                       max_multiplier=3)
                                        ]),
                                        normalize=True)
    test_5cls = Initial_dataset_loader(test_dataset_path, full=True, normalize=True)

    res_train_5cls = resampling_dataset_loader(train_dataset_path, full=True,
                                        normalize=True)
    res_test_5cls = resampling_dataset_loader(test_dataset_path, full=True, normalize=True)
    print("Loaded 5 cls")
    prev_experiments = [
    {
        "name": "SiameseCNN_4cls",
        "is_odenet": 2,
        "lr": 0.1,
        "train_dataset": train_4cls,
        "test_dataset": test_4cls,
        "labels": {
            0: "T1",
            1: "T2",
            2: "T3",
            3: "T4"
        }
    },{
        "name": "SiameseCNN_5cls",
        "is_odenet": 2,
        "lr": 0.1,
        "train_dataset": train_5cls,
        "test_dataset": test_5cls,
        "labels": {
            0: "T1",
            1: "T2",
            2: "T3",
            3: "T4",
            4: "AE"
        }
    },{
        "name": "Agumented_SiameseResNet_4cls",
        "is_odenet": 1,
        "lr": 0.1,
        "train_dataset": train_4cls,
        "test_dataset": test_4cls,
        "labels": {
            0: "T1",
            1: "T2",
            2: "T3",
            3: "T4"
        }
    }
    ]
    experiments = [
        # {
        #     "name": "Agumented_SiameseResNet_5cls",
        #     "is_odenet": 1,
        #     "lr": 0.1,
        #     "train_dataset": train_5cls,
        #     "test_dataset": test_5cls,
        #     "labels": {
        #         0: "T1",
        #         1: "T2",
        #         2: "T3",
        #         3: "T4",
        #         4: "AE"
        #     }
        # },{
        #     "name": "Agumented_SiameseODE_4cls",
        #     "is_odenet": 0,
        #     "lr": 0.1,
        #     "train_dataset": train_4cls,
        #     "test_dataset": test_4cls,
        #     "labels": {
        #         0: "T1",
        #         1: "T2",
        #         2: "T3",
        #         3: "T4"
        #     }
        # },
        # {
        #     "name": "Agumented_SiameseODE_5cls",
        #     "is_odenet": 0,
        #     "lr": 0.1,
        #     "train_dataset": train_5cls,
        #     "test_dataset": test_5cls,
        #     "labels": {
        #         0: "T1",
        #         1: "T2",
        #         2: "T3",
        #         3: "T4",
        #         4: "AE"
        #     }
        # },
        # {
        #     "name": "Resampling_SiameseResNet_4cls",
        #     "is_odenet": 1,
        #     "lr": 0.1,
        #     "train_dataset": res_train_4cls,
        #     "test_dataset": res_test_4cls,
        #     "labels": {
        #         0: "T1",
        #         1: "T2",
        #         2: "T3",
        #         3: "T4"
        #     }
        # },{
        #     "name": "Resampling_SiameseResNet_5cls",
        #     "is_odenet": 1,
        #     "lr": 0.1,
        #     "train_dataset": res_train_5cls,
        #     "test_dataset": res_test_5cls,
        #     "labels": {
        #         0: "T1",
        #         1: "T2",
        #         2: "T3",
        #         3: "T4",
        #         4: "AE"
        #     }
        # },
        # {
        #     "name": "Resampling_SiameseODE_4cls",
        #     "is_odenet": 0,
        #     "lr": 0.1,
        #     "train_dataset": res_train_4cls,
        #     "test_dataset": res_test_4cls,
        #     "labels": {
        #         0: "T1",
        #         1: "T2",
        #         2: "T3",
        #         3: "T4"
        #     }
        # },
        {
            "name": "Resampling_SiameseODE_5cls",
            "is_odenet": 0,
            "lr": 0.1,
            "train_dataset": res_train_5cls,
            "test_dataset": res_test_5cls,
            "labels": {
                0: "T1",
                1: "T2",
                2: "T3",
                3: "T4",
                4: "AE"
            }
        }
    ]

    for experiment in experiments:
        best_acc, best_f1 = trainODE(experiment["name"],
                                     (torch.utils.data.DataLoader(experiment["train_dataset"],
                                                                  batch_size=batch_size,
                                                                  shuffle=True),
                                      torch.utils.data.DataLoader(experiment["test_dataset"],
                                                                   batch_size=batch_size,
                                                                   shuffle=True)),
                                     experiment["labels"],
                                     training_length=100,
                                     type=experiment["is_odenet"],
                                     batch_size=batch_size,
                                     lr=experiment["lr"])
        df_new = pd.DataFrame(
            {
            "name": [experiment["name"]],
            "best_acc": [best_acc],
            "best_f1": [best_f1]
            }
        )
        df = df.append(df_new, ignore_index=True)
        df.to_csv("results/results_siamese.csv")
