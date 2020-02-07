import os

import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from models.AEmodel import VAE, CNNVAE
from models.RNNmodel import LSTM
from models.CNNmodel import CNN
from training_loop import Trainer, VAETrainer
from utils import Initial_dataset_loader


class Manager:
    def __init__(self, experiment_name, model, dataset, criterion, optimizer, scheduler=None, VAE=False, full=False):
        self.experiment_name = self._get_full_name(experiment_name)
        self.model = model
        self.datasets = os.path.join(os.getcwd(), "datasets", dataset)
        self.train_dataset_path = os.path.join(self.datasets, "train")
        self.train_dataset = Initial_dataset_loader(self.train_dataset_path, full=full)
        self.train_dataloader = DataLoader(self.train_dataset, 64, shuffle=True, num_workers=0)
        self.test_dataset_path = os.path.join(self.datasets, "test")
        self.test_dataset = Initial_dataset_loader(self.test_dataset_path, full=full)
        self.test_dataloader = DataLoader(self.test_dataset, 64, shuffle=True, num_workers=0)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not VAE:
            self.trainer = Trainer(experiment_name, self.model, self.train_dataloader, self.test_dataloader, criterion,
                                   optimizer, scheduler)
        else:
            self.trainer = VAETrainer(experiment_name, self.model, self.train_dataloader, self.test_dataloader,
                                      optimizer, scheduler)

    def run(self, number_of_epochs):
        self.trainer.train(number_of_epochs)

    def _get_full_name(self, name):
        max = 0
        for subdir in os.listdir(os.path.join(os.getcwd(), "experiments")):
            if name in subdir:
                var = int(subdir.split('_')[-1])
                if var > max:
                    max = var
        return name + "_" + str(max+1)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    # parser.add_argument("name", type=str, help="name of the experiment")
    # parser.add_argument("-m", "--model", type=str, help="Model to train ('FC', 'LSTM', 'VAE'")
    # parser.add_argument("-d", "--dataset", action=str, help="subfolder of the datasets folder")
    # args = parser.parse_args()
    name = "CNNVAE"
    model = CNNVAE()
    dataset = "full_corrected_dataset"
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    manager = Manager(name, model, dataset, criterion, optimizer, VAE=True)
    manager.run(500)
