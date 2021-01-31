import os

import torch
from torch import nn
import pickle
import datetime as dt
from experiment_manager import Manager
from models.CNNmodel import CNN
from models.FCmodel import FCmodel  
from models.RNNmodel import LSTMFCN, LSTMFRN
from models.SiameseModels import SiameseNeuralODE, SiameseResNet
from models.ResnetODEmodels import ODE, ResNet
from utils import resampling_dataset_loader, Memory_efficient_loader, learning_rate_with_decay
from models.Lambdamodel import LambdaResNet
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Lambda
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from metrics import best_accuracy_ml


'''
PRETRAINING
'''
# length = 15
# dataset = "Unsupervised_packed_training"
# datasets = os.path.join(os.getcwd(), "datasets", dataset)
# train_dataset_path = datasets
# train_dataset = Memory_efficient_loader(train_dataset_path, csv=False)
# experiments = [
# {
#         "model": SiameseResNet(5, ae=True),
#         "name_full": "SiameseResNet",
#         "criterion": nn.MSELoss(),
#         "optimizer": lambda x: torch.optim.Adam(x, lr=0.01, weight_decay=1e-5),
#         "scheduler": learning_rate_with_decay(0.01, 1, 1, length, boundary_epochs=[5, 10], decay_rates=[1, 0.1, 0.01]),
#         "manager": {
#             "pretraining": True,
#             "metrics": None,
#             "o2p_fcn":None,
#             "test_dataset":None,
#             "loader_size":1 ,
#             "normalize":True,
#             "nfe_logging":False,
#             "loss_preprocessing":lambda x: torch.reshape(x, [x.size()[0], -1, 2]),
#             "leading_metric":"Loss"
#         }
#     }
#     ]
'''
TRAINING
'''
length = 100
dataset = "full_siamese_dataset"
datasets = os.path.join(os.getcwd(), "datasets", dataset)
train_dataset_path = os.path.join(datasets, "train")
mapping_path = os.path.join(os.getcwd(), "datasets", "full_siamese_dataset_to_full_extended_dataset_mapping.csv")
labels_path = os.path.join(os.getcwd(), "datasets", "train_corrections.csv")
train_dataset = resampling_dataset_loader(train_dataset_path, siamese=True,
                    multilabel_mapping_path=mapping_path, multilabel_labels_path=labels_path, nucci=False)
test_dataset_path = os.path.join(datasets, "test")
test_dataset = resampling_dataset_loader(test_dataset_path, siamese=True,
                    multilabel_mapping_path=mapping_path, multilabel_labels_path=labels_path, nucci=False)

experiments = [
    # {
    #     "model": SiameseResNet(5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "SiameseResNet",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },

    # {
    #     "model": FCmodel(180, 5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "FullyConnected",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },

    # {
    #     "model": LSTMFRN(180, 5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "LSTMFRN",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": False,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    {
        "model": ResNet(5),
        "pretrained": False,
        "pretraining_path": None,
        "name_full": "ResNet",
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
        "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": False,
            "metrics": {
                "Accuracy": accuracy_score,
                "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
            },
            "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
            "test_dataset": test_dataset,
            "loader_size": 256,
            "normalize": True,
            "nfe_logging": False,
            "loss_preprocessing": None,
            "leading_metric": "Accuracy",
            "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
        },
        "dataset": {
            "siamese": False,
            "multilabel": False,
            "include_artificial_ae": True,
            "transforms": None
        }
    },
    # {
    #     "model": CNN(5),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "CNN",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": FCmodel(360, 5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "FullyConnected_DualChannel",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [torch.cat(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },

    # {
    #     "model": LSTMFRN(180, 5, in_channels=2, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "LSTMFRN_DualChannel",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ResNet(5, in_channels=2),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ResNet_DualChannel",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": CNN(5, channels=[2, 32, 64, 64]),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "CNN_DualChannel",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": SiameseResNet(5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "SiameseResNet_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },

    # {
    #     "model": FCmodel(180, 5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "FullyConnected_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },

    # {
    #     "model": LSTMFCN(180, 5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "LSTMFCN_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ResNet(5),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ResNet_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": CNN(5),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "CNN_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": FCmodel(360, 5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "FullyConnected_DualChannel_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [torch.cat(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },

    # {
    #     "model": LSTMFCN(180, 5, in_channels=2, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "LSTMFCN_DualChannel_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ResNet(5, in_channels=2),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ResNet_DualChannel_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": CNN(5, channels=[2, 32, 64, 64]),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "CNN_DualChannel_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": SiameseNeuralODE(5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "SiameseNeuralODE",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": True,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": SiameseNeuralODE(5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "SiameseNeuralODE_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": True,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": None
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ODE(5, in_channels=2),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ODE_DualChannel_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": True,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": True,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ODE(5),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ODE_Multilabel",
    #     "criterion": nn.BCELoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="macro"),
    #             "Jaccard": lambda labels, preds: jaccard_score(labels, preds, average="macro"),
    #             "Best_accuracy": best_accuracy_ml
    #         },
    #         "o2p_fcn": lambda x: np.where(torch.sigmoid(x.cpu().detach()).numpy() >= 0.5, np.ones(x.shape), np.zeros(x.shape)),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": True,
    #         "loss_preprocessing": lambda x: torch.sigmoid(x),
    #         "leading_metric": "F1_Score",
    #         "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": True,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ODE(5),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ODE",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": True,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # },
    # {
    #     "model": ODE(5, in_channels=2),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "ODE_DualChannel",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": True,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [torch.stack(x, dim=1)]
    #     },
    #     "dataset": {
    #         "siamese": True,
    #         "multilabel": False,
    #         "include_artificial_ae": True,
    #         "transforms": None
    #     }
    # }
    # {
    #     "model": LambdaResNet(5, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "LambdaResNet",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": False,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # }
    # {
    #     "model": FCmodel(20, 5, hidden1 = 16, hidden2 = 8, ae=False),
    #     "pretrained": False,
    #     "pretraining_path": None,
    #     "name_full": "Test",
    #     "criterion": nn.CrossEntropyLoss(),
    #     "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
    #     "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
    #     "manager": {
    #         "pretraining": False,
    #         "metrics": {
    #             "Accuracy": accuracy_score,
    #             "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
    #         },
    #         "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
    #         "test_dataset": test_dataset,
    #         "loader_size": 256,
    #         "normalize": True,
    #         "nfe_logging": False,
    #         "loss_preprocessing": None,
    #         "leading_metric": "Accuracy",
    #         "input_preprocessing": None,
    #     },
    #     "dataset": {
    #         "siamese": False,
    #         "multilabel": False,
    #         "include_artificial_ae": False,
    #         "transforms": None
    #     }
    # },
    {
        "model": ResNet(5, depth=5),
        "pretrained": False,
        "pretraining_path": None,
        "name_full": "ResNet_d5",
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
        "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": False,
            "metrics": {
                "Accuracy": accuracy_score,
                "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
            },
            "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
            "test_dataset": test_dataset,
            "loader_size": 256,
            "normalize": True,
            "nfe_logging": False,
            "loss_preprocessing": None,
            "leading_metric": "Accuracy",
            "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
        },
        "dataset": {
            "siamese": False,
            "multilabel": False,
            "include_artificial_ae": True,
            "transforms": None
        }
    },
    {
        "model": ResNet(5, depth=4),
        "pretrained": False,
        "pretraining_path": None,
        "name_full": "ResNet_d4",
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
        "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": False,
            "metrics": {
                "Accuracy": accuracy_score,
                "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
            },
            "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
            "test_dataset": test_dataset,
            "loader_size": 256,
            "normalize": True,
            "nfe_logging": False,
            "loss_preprocessing": None,
            "leading_metric": "Accuracy",
            "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
        },
        "dataset": {
            "siamese": False,
            "multilabel": False,
            "include_artificial_ae": True,
            "transforms": None
        }
    },
    {
        "model": ResNet(5, depth=3),
        "pretrained": False,
        "pretraining_path": None,
        "name_full": "ResNet_d3",
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
        "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": False,
            "metrics": {
                "Accuracy": accuracy_score,
                "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
            },
            "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
            "test_dataset": test_dataset,
            "loader_size": 256,
            "normalize": True,
            "nfe_logging": False,
            "loss_preprocessing": None,
            "leading_metric": "Accuracy",
            "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
        },
        "dataset": {
            "siamese": False,
            "multilabel": False,
            "include_artificial_ae": True,
            "transforms": None
        }
    },
    {
        "model": ResNet(5, depth=2),
        "pretrained": False,
        "pretraining_path": None,
        "name_full": "ResNet_d2",
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
        "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": False,
            "metrics": {
                "Accuracy": accuracy_score,
                "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
            },
            "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
            "test_dataset": test_dataset,
            "loader_size": 256,
            "normalize": True,
            "nfe_logging": False,
            "loss_preprocessing": None,
            "leading_metric": "Accuracy",
            "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
        },
        "dataset": {
            "siamese": False,
            "multilabel": False,
            "include_artificial_ae": True,
            "transforms": None
        }
    },
    {
        "model": ResNet(5, depth=1),
        "pretrained": False,
        "pretraining_path": None,
        "name_full": "ResNet_d1",
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.95, nesterov=True),
        "scheduler": learning_rate_with_decay(0.01, 256, 256, length, boundary_epochs=[33, 66], decay_rates=[1, 0.1, 0.01]),
        "manager": {
            "pretraining": False,
            "metrics": {
                "Accuracy": accuracy_score,
                "F1_Score": lambda labels, preds: f1_score(labels, preds, average="weighted")
            },
            "o2p_fcn": lambda x: np.argmax(x.cpu().detach().numpy(), axis=1),
            "test_dataset": test_dataset,
            "loader_size": 256,
            "normalize": True,
            "nfe_logging": False,
            "loss_preprocessing": None,
            "leading_metric": "Accuracy",
            "input_preprocessing": lambda x: [a.unsqueeze(1) for a in x]
        },
        "dataset": {
            "siamese": False,
            "multilabel": False,
            "include_artificial_ae": True,
            "transforms": None
        }
    },
    ]
    

if __name__ == "__main__":
    batch_name = str(dt.date.today()) + "_Tryout_"
    log = []
    reproducability = True

    if reproducability:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    cols = ["name", "#Params", "Loss"]
    result_dataframe = pd.DataFrame(columns=cols)
    for experiment in experiments:
        # try:
        name_full = batch_name + experiment["name_full"]
        model = experiment["model"]
        train_dataset.set_parameters(siamese=experiment["dataset"]["siamese"],
                            multilabel=experiment["dataset"]["multilabel"],
                            include_artificial_ae=experiment["dataset"]["include_artificial_ae"],
                            transforms=experiment["dataset"]["transforms"])
        test_dataset.set_parameters(siamese=experiment["dataset"]["siamese"],
                            multilabel=experiment["dataset"]["multilabel"],
                            include_artificial_ae=experiment["dataset"]["include_artificial_ae"],
                            transforms=experiment["dataset"]["transforms"])
        if experiment["pretrained"]:
            model.load_state_dict(torch.load(experiment["pretraining_path"])['state_dict'])
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
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
                        leading_metric=experiment["manager"]["leading_metric"],
                        input_preprocessing=experiment["manager"]["input_preprocessing"]
                        )
        manager.run(length)
        result_dataframe = result_dataframe.append(pd.DataFrame(manager.get_results()), ignore_index=True)
        result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name + ".csv"))
        # except Exception as e:
        #     log.append("failed model " + name_full + "\n Error message: " + repr(e))
        #     continue
    
    print(log)
    # with open(os.path.join(os.getcwd(), "results", batch_name + ".pkl"), 'wb') as outfile:
    #     pickle.dump(experiments, outfile)
