import os

import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from models.CNNmodel import CNN2d
from models.RNNmodel import LSTM
from training_loop import Trainer, PreTrainer
from utils import Initial_dataset_loader, get_fourier_coeff, PlotToImage, ShortenOrElongateTransform
import pandas as pd
import numpy as np
'''
class_dict, metrics, output_to_pred_fcn, loss_preprocessing=None,
leading_metric=None, scheduler=None, nfe_logging = False
'''

class Manager:
    def __init__(self, experiment_name, model, train_dataset, criterion,
                optimizer, metrics = None, output_to_pred_fcn = None, test_dataset=None,
                scheduler=None, loader_size=256, normalize=True, nfe_logging = False,
                loss_preprocessing = None, leading_metric = None,
                class_dict = {0: "T1", 1: "T2", 2: "T3", 3: "T4", 4: "AE"},
                pretraining = False):

        self.experiment_name = self._get_full_name(experiment_name)
        self.model = model
        self.metrics = metrics
        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(self.train_dataset, loader_size, shuffle=False, num_workers=0)

        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_dataloader = DataLoader(self.test_dataset, loader_size, shuffle=False, num_workers=0)

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.leading_metric = leading_metric
        if not pretraining:
            self.trainer = Trainer(self.experiment_name, self.model, self.train_dataloader,
                            self.test_dataloader, criterion, optimizer, class_dict,
                            metrics, output_to_pred_fcn, loss_preprocessing,
                            leading_metric, scheduler, nfe_logging)
        else:
            self.trainer = PreTrainer(self.experiment_name, self.model, self.train_dataloader,
                                    criterion, optimizer, loss_preprocessing, scheduler)

    def run(self, number_of_epochs):
        print("Starting experiment - " + self.experiment_name)
        self.best_leading_metric, self.other_metrics = self.trainer.train(number_of_epochs)

    def get_results(self):
        results_dict = {
            "name": self.experiment_name,
            "#Params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            self.best_leading_metric[0]: self.best_leading_metric[1],
        }

        for i in range(len(self.other_metrics)):
            if self.other_metrics[i] is not None:
                results_dict[self.other_metrics[i][0]] = self.other_metrics[i][1]
        return results_dict

    def _get_full_name(self, name):
        max = 0
        for subdir in os.listdir(os.path.join(os.getcwd(), "experiments")):
            if name in subdir:
                var = int(subdir.split('_')[-1])
                if var > max:
                    max = var
        return name + "_" + str(max+1)


if __name__ == "__main__":
    name = "Cnn2d"
    model = CNN2d((180, 180), 5)
    dataset = "full_corrected_dataset"
    criterion = nn.CrossEntropyLoss()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    manager = Manager(name, model, dataset, criterion, optimizer, full=True, loader_size=32,
                      scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 100, 140]),
                      image_size=(180, 180))
    manager.run(500)
