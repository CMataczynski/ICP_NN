import os

import torch
from torch import nn

import datetime as dt
from experiment_manager import Manager
from models.AEmodel import VAE, CNNVAE, AE, CNNAE
from models.CNNmodel import CNN, CNN2d
from models.FCmodel import FCmodel
from models.RNNmodel import LSTM, GRU, LSTMFCN
from models.SiameseModels import SiameseNeuralODE, SiameseResNet
from models.ResnetODEmodels import ODE, ResNet
from utils import resampling_dataset_loader, Memory_efficient_loader, learning_rate_with_decay
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Lambda
from ODETrainingLoop_Sigmoid import trainODE
from SiameseTrainingLoop import trainODE as trainSiameseODE


length = 15
# rootdir = os.path.join(os.getcwd(), 'experiments')
dataset = "Unsupervised_training_dataset"
datasets = os.path.join(os.getcwd(), "datasets", dataset)
# train_dataset_path = os.path.join(datasets, "train")
train_dataset_path = datasets
train_dataset = Memory_efficient_loader(train_dataset_path)
# weights = train_dataset.get_class_weights()
# train_dataset = resampling_dataset_loader(train_dataset_path, full=True)
# weights_5cls = train_dataset.get_class_weights()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weights_5cls = weights_5cls.to(device)
# weights = weights.to(device)
# train_dataset = None

experiments = [
{
        "model": SiameseResNet(5, ae=True),
        "name_full": "SiameseResNet",
        "criterion": nn.MSELoss(),
        "optimizer": lambda x: torch.optim.Adam(x, lr=0.01, weight_decay=1e-5),
        "scheduler": learning_rate_with_decay(0.01, 3000, 3000, length, boundary_epochs=[5, 10], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": True,
            "metrics": None,
            "o2p_fcn":None,
            "test_dataset":None,
            "loader_size":3000 ,
            "normalize":True,
            "nfe_logging":False,
            "loss_preprocessing":lambda x: torch.reshape(x, [x.size()[0], -1, 2]),
            "leading_metric":"Loss"
        }
    }
    ]


if __name__ == "__main__":
    batch_name = str(dt.date.today())+"_pretraining_"
    log = []
    reproducability = True

    if reproducability:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic   = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    cols = ["Nazwa", "Parametry", "Loss", ""]
    result_dataframe = pd.DataFrame(columns=cols)
    for experiment in experiments:
        model = experiment["model"]
        name_full = batch_name + experiment["name_full"]
        # try:
        criterion = experiment["criterion"]
        optimizer = experiment["optimizer"]
        scheduler = experiment["scheduler"]
        manager = Manager(name_full, model, train_dataset, criterion, optimizer,
                      pretraining=experiment["manager"]["pretraining"],
                      metrics=experiment["manager"]["metrics"],
                      output_to_pred_fcn=experiment["manager"]["o2p_fcn"],
                      test_dataset=experiment["manager"]["test_dataset"],
                      scheduler=scheduler,
                      loader_size=experiment["manager"]["loader_size"],
                      normalize=experiment["manager"]["normalize"],
                      nfe_logging=experiment["manager"]["nfe_logging"],
                      loss_preprocessing=experiment["manager"]["loss_preprocessing"],
                      leading_metric=experiment["manager"]["leading_metric"]
                      )
        # print("Starting experiment {}".format(name_full))
        manager.run(length)
        result_dataframe = result_dataframe.append(pd.DataFrame(manager.get_results()), ignore_index=True)
        result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name + ".csv"), sep=';', decimal=',')
        # except:
        #     log.append("failed model " + name_full)
        #     continue
