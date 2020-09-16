import os

import torch
from torch import nn

import datetime as dt
from experiment_manager import Manager
from models.AEmodel import VAE, CNNVAE, AE, CNNAE
from models.CNNmodel import CNN, CNN2d
from models.FCmodel import FCmodel
from models.RNNmodel import LSTM, GRU, LSTMFCN
from utils import Initial_dataset_loader, get_fourier_coeff, ShortenOrElongateTransform, transform_fourier, PlotToImage, resampling_dataset_loader
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Lambda
from ODETrainingLoop_Sigmoid import trainODE
from SiameseTrainingLoop import trainODE as trainSiameseODE


run_ODE = True
length = 150
# rootdir = os.path.join(os.getcwd(), 'experiments')
# dataset = "full_extended_dataset"
# datasets = os.path.join(os.getcwd(), "datasets", dataset)
# train_dataset_path = os.path.join(datasets, "train")
# train_dataset = resampling_dataset_loader(train_dataset_path)
# weights = train_dataset.get_class_weights()
# train_dataset = resampling_dataset_loader(train_dataset_path, full=True)
# weights_5cls = train_dataset.get_class_weights()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weights_5cls = weights_5cls.to(device)
# weights = weights.to(device)
# train_dataset = None

# experiments = [
# # {
# #         "model": CNN(178, 5),
# #         "name_full": "NormalizedFourier_CNN_5cls_weighted",
# #         "criterion": nn.CrossEntropyLoss(weights_5cls),
# #         "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
# #         "scheduler": lambda x: torch.optim.lr_scheduler.MultiStepLR(x, [60, 90, 130]),
# #         "manager": {
# #             "VAE": False,
# #             "full": True,
# #             "ortho": get_fourier_coeff,
# #             "transforms": None,
# #             "test_transforms": None,
# #             'image_size': None,
# #             "loader_size": 64,
# #             "normalize": True
# #         }
# #     },
# {
#         "model": LSTMFCN(180, 5),
#         "name_full": "RAW_LSTM_FCN_5cls_weighted",
#         "criterion": nn.CrossEntropyLoss(weights_5cls),
#         "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
#         "scheduler": lambda x: torch.optim.lr_scheduler.MultiStepLR(x, [60, 90, 130]),
#         "manager": {
#             "VAE": False,
#             "full": True,
#             "ortho": None,
#             "transforms": Compose([
#                 ShortenOrElongateTransform(min_length=32,
#                                            max_length=180,
#                                            probability=0.7,
#                                            max_multiplier=3)
#             ]),
#             "test_transforms": None,
#             "image_size": None,
#             "loader_size": 64,
#             "normalize": True
#         }
#     },
#     {
#         "model": LSTMFCN(180, 4),
#         "name_full": "RAW_LSTM_FCN_4cls_weighted",
#         "criterion": nn.CrossEntropyLoss(weights),
#         "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
#         "scheduler": lambda x: torch.optim.lr_scheduler.MultiStepLR(x, [60, 90, 130]),
#         "manager": {
#             "VAE": False,
#             "full": False,
#             "ortho": None,
#             "transforms": Compose([
#                 ShortenOrElongateTransform(min_length=32,
#                                            max_length=180,
#                                            probability=0.7,
#                                            max_multiplier=3)
#             ]),
#             "test_transforms": None,
#             "image_size": None,
#             "loader_size": 64,
#             "normalize": True
#         }
#     }]
#     # {
#     #     "model": CNN(178, 4),
#     #     "name_full": "NormalizedFourier_CNN_4cls_weighted",
#     #     "criterion": nn.CrossEntropyLoss(weights),
#     #     "optimizer": lambda x: torch.optim.Adam(x, lr=0.01),
#     #     "scheduler": lambda x: torch.optim.lr_scheduler.MultiStepLR(x, [60, 90, 130]),
#     #     "manager": {
#     #         "VAE": False,
#     #         "full": False,
#     #         "ortho": get_fourier_coeff,
#     #         "transforms": None,
#     #         "test_transforms": None,
#     #         'image_size': None,
#     #         "loader_size": 64,
#     #         "normalize": True
#     #     }
#     # }


if __name__ == "__main__":
    batch_name = str(dt.date.today())+"_resampling_ml"
    log = []
    reproducability = True

    if reproducability:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    cols = ["Nazwa", "Parametry", "Accuracy [%]", "F1 Score"]
    result_dataframe = pd.DataFrame(columns=cols)
    # for experiment in experiments:
    #     model = experiment["model"]
    #     print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    #     name_full = batch_name + experiment["name_full"]
    #     try:
    #         criterion = experiment["criterion"]
    #         optimizer = experiment["optimizer"](model.parameters())
    #         scheduler = experiment["scheduler"](optimizer)
    #         manager = Manager(name_full, model, dataset, criterion, optimizer,
    #                       VAE=experiment["manager"]["VAE"],
    #                       full=experiment["manager"]["full"],
    #                       ortho=experiment["manager"]["ortho"],
    #                       transforms=experiment["manager"]["transforms"],
    #                       test_transforms=experiment["manager"]["test_transforms"],
    #                       loader_size=experiment["manager"]["loader_size"],
    #                       normalize=experiment["manager"]["normalize"],
    #                       image_size=experiment["manager"]["image_size"],
    #                       scheduler=scheduler)
    #         manager.run(length)
    #         result_dataframe = result_dataframe.append(pd.DataFrame(manager.get_results(), columns=cols), ignore_index=True)
    #         result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name + ".csv"), sep=';', decimal=',')
    #     except:
    #         log.append("failed model " + name_full)
    #         continue
    if run_ODE:
        ODE_runs = [
            # (False, True, 256, False, None, batch_name + "SigmoidResNet_5cls"),
            # (False, True, 256, False, 2000, batch_name + "2kae_SigmoidResNet_5cls"),
            # (False, True, 256, True, None, batch_name + "Multilabel_SigmoidResNet_5cls"),
            (False, True, 256, True, 2000, batch_name + "2kae_Multilabel_SigmoidResNet_5cls")
        ]
    else:
        ODE_runs = []

    for run in ODE_runs:
        ODE_run = trainODE(run[0], run[1], run[2], run[3], run[4])
        result_dataframe = result_dataframe.append(
            pd.DataFrame([[str(run[5]), "-"] + ODE_run], columns=cols), ignore_index=True
        )
        result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name+".csv"), sep=';', decimal=',')

    ml_path_train = os.path.join(os.getcwd(), "datasets", "train_corrections.csv")
    ml_path_test = os.path.join(os.getcwd(), "datasets", "test_corrections.csv")
    ml_mapping_path = os.path.join(os.getcwd(), "datasets", "full_siamese_dataset_to_full_extended_dataset_mapping.csv")
    res_ae_train_5cls = resampling_dataset_loader(train_dataset_path, full=True,
                                        normalize=True, siamese=True,
                                        artificial_ae_examples=2000, multilabel_labels_path = ml_path_train,
                                        multilabel_mapping_path = ml_mapping_path)
    res_test_5cls = resampling_dataset_loader(test_dataset_path, full=True, normalize=True,
                                        siamese=True, multilabel_labels_path = ml_path_test,
                                        multilabel_mapping_path = ml_mapping_path)
    res_train_5cls = resampling_dataset_loader(train_dataset_path, full=True,
                                        normalize=True, siamese=True, multilabel_labels_path = ml_path_train,
                                        multilabel_mapping_path = ml_mapping_path)
    res_mlae_train_5cls = resampling_dataset_loader(train_dataset_path, full=True,
                                        normalize=True, siamese=True,
                                        artificial_ae_examples=2000, multilabel_labels_path = ml_path_train,
                                        multilabel_mapping_path = ml_mapping_path,
                                        multilabel=True)
    res_ml_test_5cls = resampling_dataset_loader(test_dataset_path, full=True, normalize=True,
                                        siamese=True, multilabel_labels_path = ml_path_test,
                                        multilabel_mapping_path = ml_mapping_path,
                                        multilabel=True)
    res_ml_train_5cls = resampling_dataset_loader(train_dataset_path, full=True,
                                        normalize=True, siamese=True, multilabel_labels_path = ml_path_train,
                                        multilabel_mapping_path = ml_mapping_path,
                                        multilabel=True)


    experiments = [
        {
            "name": "Resampling_SigmoidSiameseResNet_5cls_artificial",
            "is_odenet": 1,
            "lr": 0.1,
            "train_dataset": res_ae_train_5cls,
            "test_dataset": res_test_5cls,
            "labels": {
                0: "T1",
                1: "T2",
                2: "T3",
                3: "T4",
                4: "AE"
            },
            "sigmoid": True,
            "multilabel": False
        },{
            "name": "Resampling_SigmoidSiameseResNet_5cls_artificial_ml",
            "is_odenet": 1,
            "lr": 0.1,
            "train_dataset": res_mlae_train_5cls,
            "test_dataset": res_ml_test_5cls,
            "labels": {
                0: "T1",
                1: "T2",
                2: "T3",
                3: "T4",
                4: "AE"
            },
            "sigmoid": True,
            "multilabel": True
        },{
            "name": "Resampling_SigmoidSiameseResNet_5cls",
            "is_odenet": 1,
            "lr": 0.1,
            "train_dataset": res_train_5cls,
            "test_dataset": res_test_5cls,
            "labels": {
                0: "T1",
                1: "T2",
                2: "T3",
                3: "T4",
                4: "AE"
            },
            "sigmoid": True,
            "multilabel": False
        },{
            "name": "Resampling_SigmoidSiameseResNet_5cls_ml",
            "is_odenet": 1,
            "lr": 0.1,
            "train_dataset": res_ml_train_5cls,
            "test_dataset": res_ml_test_5cls,
            "labels": {
                0: "T1",
                1: "T2",
                2: "T3",
                3: "T4",
                4: "AE"
            },
            "sigmoid": True,
            "multilabel": True
        }
    ]
    for experiment in experiments:
        best_acc, best_f1 = trainSiameseODE(experiment["name"],
                                     (torch.utils.data.DataLoader(experiment["train_dataset"],
                                                                  batch_size=batch_size,
                                                                  shuffle=True),
                                      torch.utils.data.DataLoader(experiment["test_dataset"],
                                                                   batch_size=batch_size,
                                                                   shuffle=True)),
                                     experiment["labels"],
                                     training_length=100,
                                     type=experiment["is_odenet"],
                                     batch_size=256,
                                     lr=experiment["lr"],
                                     multilabel=experiment["multilabel"],
                                     sigmoid=experiment["sigmoid"])
        df_new = pd.DataFrame(
            {
            "name": [experiment["name"]],
            "best_acc": [best_acc],
            "best_f1": [best_f1]
            }
        )
        df = df.append(df_new, ignore_index=True)
        df.to_csv("results/results_siamese.csv")
    print(log)
