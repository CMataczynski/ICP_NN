import os

import torch
from torch import nn

from experiment_manager import Manager
from models.AEmodel import VAE, CNNVAE, AE, CNNAE
from models.CNNmodel import CNN, CNN2d
from models.FCmodel import FCmodel
from models.RNNmodel import LSTM, GRU, LSTMFCN
from utils import Initial_dataset_loader, get_fourier_coeff, ShortenOrElongateTransform, transform_fourier, PlotToImage
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Lambda
from utils import TransformToEmd
from ODETrainingLoop import trainODE


length = 150
rootdir = os.path.join(os.getcwd(), 'experiments')
dataset = "full_splitted_dataset"
datasets = os.path.join(os.getcwd(), "datasets", dataset)
train_dataset_path = os.path.join(datasets, "train")
train_dataset = Initial_dataset_loader(train_dataset_path)
weights = train_dataset.get_class_weights()
train_dataset = Initial_dataset_loader(train_dataset_path, full=True)
weights_5cls = train_dataset.get_class_weights()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_5cls = weights_5cls.to(device)
weights = weights.to(device)
train_dataset = None
ortho_emd = TransformToEmd()

experiments = [
{
        "model": LSTMFCN(180, 5),
        "name_full": "RAW_LSTM_FCN_5cls_weighted",
        "criterion": nn.CrossEntropyLoss(weights_5cls),
        "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
        "manager": {
            "VAE": False,
            "full": True,
            "ortho": None,
            "transforms": Compose([
                ShortenOrElongateTransform(min_length=32,
                                           max_length=180,
                                           probability=0.7,
                                           max_multiplier=3)
            ]),
            "test_transforms": None,
            "image_size": None,
            "loader_size": 64,
            "normalize": True
        }
    },
    {
        "model": LSTMFCN(180, 4),
        "name_full": "RAW_LSTM_FCN_4cls_weighted",
        "criterion": nn.CrossEntropyLoss(weights),
        "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
        "manager": {
            "VAE": False,
            "full": False,
            "ortho": None,
            "transforms": Compose([
                ShortenOrElongateTransform(min_length=32,
                                           max_length=180,
                                           probability=0.7,
                                           max_multiplier=3)
            ]),
            "test_transforms": None,
            "image_size": None,
            "loader_size": 64,
            "normalize": True
        }
    },
    {
        "model": CNN(178, 5),
        "name_full": "NormalizedFourier_CNN_5cls_weighted",
        "criterion": nn.CrossEntropyLoss(weights_5cls),
        "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
        "manager": {
            "VAE": False,
            "full": True,
            "ortho": get_fourier_coeff,
            "transforms": None,
            "test_transforms": None,
            'image_size': None,
            "loader_size": 64,
            "normalize": True
        }
    },
    {
        "model": CNN(178, 4),
        "name_full": "NormalizedFourier_CNN_4cls_weighted",
        "criterion": nn.CrossEntropyLoss(weights),
        "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
        "manager": {
            "VAE": False,
            "full": False,
            "ortho": get_fourier_coeff,
            "transforms": None,
            "test_transforms": None,
            'image_size': None,
            "loader_size": 64,
            "normalize": True
        }
    }
]


if __name__ == "__main__":
    batch_name = "10.03"
    log = []
    reproducability = True
    
    if reproducability:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    cols = ["Nazwa", "Parametry", "Accuracy [%]", "F1 Score"]
    result_dataframe = pd.DataFrame(columns=cols)
    for experiment in experiments:
        model = experiment["model"]
        name_full = experiment["name_full"]
        try:
            criterion = experiment["criterion"]
            optimizer = experiment["optimizer"](model.parameters())
            manager = Manager(name_full, model, dataset, criterion, optimizer,
                          VAE=experiment["manager"]["VAE"],
                          full=experiment["manager"]["full"],
                          ortho=experiment["manager"]["ortho"],
                          transforms=experiment["manager"]["transforms"],
                          test_transforms=experiment["manager"]["test_transforms"],
                          loader_size=experiment["manager"]["loader_size"],
                          normalize=experiment["manager"]["normalize"],
                          image_size=experiment["manager"]["image_size"])
            manager.run(length)
            result_dataframe = result_dataframe.append(pd.DataFrame(manager.get_results(), columns=cols), ignore_index=True)
            result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name + ".csv"), sep=';', decimal=',')
        except:
            log.append("failed model " + name_full)
            continue

    ODE_runs = [
        (False, True, 128, "ResNet_5cls"),
        (False, False, 128, "ResNet_4cls"),
        (True, True, 128, "ODE_5cls"),
        (True, False, 128, "ODE_4cls")
    ]

    for run in ODE_runs:
        ODE_run = trainODE(run[0], run[1], run[2])
        result_dataframe = result_dataframe.append(
            pd.DataFrame([[str(run[3]), "-"] + ODE_run], columns=cols), ignore_index=True
        )
        result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name+".csv"), sep=';', decimal=',')
    print(log)
