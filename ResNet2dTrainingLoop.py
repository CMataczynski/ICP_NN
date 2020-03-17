import os
import argparse
import logging
import time
import numpy as np
import torch
import datetime as dt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import plot_confusion_matrix
import torchvision.datasets as datasets
from sklearn.metrics import f1_score
import torchvision.transforms as transforms
from utils import Initial_dataset_loader, ShortenOrElongateTransform
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
# parser.add_argument('--tol', type=float, default=1e-3)
# parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
# parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
# parser.add_argument('--nepochs', type=int, default=160)
# parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
# parser.add_argument('--lr', type=float, default=0.1)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--test_batch_size', type=int, default=1000)
#
# parser.add_argument('--save', type=str, default='./experiment1')
# parser.add_argument('--debug', action='store_true')
# parser.add_argument('--gpu', type=int, default=0)
# args = parser.parse_args()
#
# if args.adjoint:
from torchdiffeq import odeint_adjoint as odeint
lr = 0.1
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
batch_size = 8

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-5, atol=1e-5)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


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


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    labs = []
    preds = []
    for data in dataset_loader:
        x = data['image'].float().to(device)
        y = one_hot(np.array(data['label'].numpy()), 5)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        labs += data['label'].tolist()
        preds += predicted_class.tolist()
        total_correct += np.sum(predicted_class == target_class)
    f1 = f1_score(labs, preds, average='weighted')
    return total_correct / len(dataset_loader.dataset), f1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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


def trainODE(is_odenet=True, full=True, batch_size=256):
    if is_odenet:
        name = "ODE"
    else:
        name = "Resnet"
    name = str(dt.date.today()) + "_" + name

    if full:
        name = name + "_5cls"
    else:
        name = name + "_4cls"

    max = 0
    for subdir in os.listdir(os.path.join(os.getcwd(), "experiments")):
        if name in subdir:
            var = int(subdir.split('_')[-1])
            if var > max:
                max = var
    name = name + "_" + str(max+1)
    writer = SummaryWriter(log_dir='experiments/' + str(name))
    makedirs(os.path.join(os.getcwd(), "experiments", name))
    downsampling_layers = [
        nn.Conv2d(1, 32, 3, 1),
        ResBlock(32, 32, stride=2, downsample=conv1x1(32, 32, 2)),
        ResBlock(32, 32, stride=2, downsample=conv1x1(32, 32, 2)),
    ]

    feature_layers = [ODEBlock(ODEfunc(32))] if is_odenet else [ResBlock(32, 32) for _ in range(3)]
    fc_layers = [norm(32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(2), Flatten(), nn.Dropout(0.6), nn.Linear(128, 5)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    datasets = os.path.join(os.getcwd(), "datasets", "full_splitted_dataset")
    train_dataset_path = os.path.join(datasets, "train")
    train_dataset = Initial_dataset_loader(train_dataset_path, full=full, image_size=(180,180))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    test_dataset_path = os.path.join(datasets, "test")
    test_dataset = Initial_dataset_loader(test_dataset_path, full=full, image_size=(180,180))
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        batch_size, batch_denom=batch_size, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_acc = 0
    best_f1 = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    running_loss = 0.0
    for itr in range(150 * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)
        model.train()
        optimizer.zero_grad()
        dct = data_gen.__next__()
        x = dct["image"].float()
        print(x.shape)
        y = dct["label"]
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print((i + epoch * examples )% 10, loss.item(), running_loss)
        if itr % 10 == 9:
            writer.add_scalar("Loss/train", running_loss / 10, itr)
            running_loss = 0.0
        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                model.eval()
                val_acc, f1 = accuracy(model, test_loader)
                if f1 > best_f1:
                    torch.save({'state_dict': model.state_dict()}, os.path.join(os.getcwd(),
                                                                                              "experiments", name,
                                                                                              'model_1.pth'))
                    best_f1 = f1
                    best_acc = val_acc
                writer.add_scalar("Accuracy/test", val_acc, itr//batches_per_epoch)
                writer.add_scalar("F1_score/test", f1, itr//batches_per_epoch)
                print(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Test Acc {:.4f} | Test F1 {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, val_acc, f1
                    )
                )

    labs = []
    preds = []
    for data in test_loader:
        x = data['image'].float().to(device)
        y = data['label'].tolist()
        labs += y
        outputs = model(x)
        predicted = torch.max(outputs, 1).indices
        preds += predicted.tolist()

    class_dict = {
        0: "T1",
        1: "T2",
        2: "T3",
        3: "T4",
        4: "A+E"
    }
    labs = [class_dict[a] for a in labs]
    preds = [class_dict[a] for a in preds]
    writer.add_figure(name + " - Confusion Matrix",
                      plot_confusion_matrix(labs, preds, ["T1", "T2", "T3", "T4", "A+E"]))
    writer.close()
    return [best_acc, best_f1]

if __name__ == "__main__":
    trainODE(is_odenet=False)