import argparse
import os
from torch import nn
import torch.optim
from utils import Initial_dataset_loader
from torch.utils.data import DataLoader
from training_loop import Trainer, VAETrainer
from models.FCmodel import FCmodel
from models.RNNmodel import LSTM
from models.AEmodel import VAE


class Manager:
    def __init__(self, experiment_name, model, dataset, criterion, optimizer, scheduler=None, VAE = False):
        self.experiment_name = experiment_name
        self.model = model
        self.datasets = os.path.join(os.getcwd(), "datasets", dataset)
        self.train_dataset = Initial_dataset_loader(self.datasets)
        self.train_dataloader = DataLoader(self.train_dataset, 4, shuffle=True, num_workers=0)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not VAE:
            self.trainer = Trainer(experiment_name,self.model, self.train_dataloader, self.train_dataloader, criterion, optimizer, scheduler)
        else:
            self.trainer = VAETrainer(experiment_name, self.model, self.train_dataloader, self.train_dataloader,
                                   optimizer, scheduler)
    def run(self, number_of_epochs):
        self.trainer.train(number_of_epochs)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    # parser.add_argument("name", type=str, help="name of the experiment")
    # parser.add_argument("-m", "--model", type=str, help="Model to train ('FC', 'LSTM', 'VAE'")
    # parser.add_argument("-d", "--dataset", action=str, help="subfolder of the datasets folder")
    # args = parser.parse_args()
    name = "VAE"
    rootdir = os.path.join(os.getcwd(), 'experiments')
    max = 0
    for subdir in os.listdir(rootdir):
        if name in subdir:
            var = int(subdir.split('_')[-1])
            if var > max:
                max = var
    name = name + "_" + str(max+1)
    
    model = VAE()
    dataset = "initial_dataset"
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    manager = Manager(name, model, dataset, criterion, optimizer, VAE=True)
    manager.run(1000)