import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import Initial_dataset_loader
import tqdm
import pickle, copyreg
import numpy as np
from sklearn.svm import SVC
import os

class Trainer:
    def __init__(self,name, network, train_dataloader, test_dataloader, criterion, optimizer, scheduler=None):
        self.name = name
        self.net = network
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, number_of_epochs):
        writer = SummaryWriter(log_dir='experiments/'+str(self.name))
        device = self.device
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader
        net = self.net.double()
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        net.to(device)

        # criterion = self.criterion
        # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 250], gamma=0.1)

        for epoch in tqdm.tqdm(range(number_of_epochs)):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs = data['image'].double().to(device)
                labels = data['label'].to(device)

                optimizer.zero_grad()
                outputs = net(inputs).double()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i + epoch * len(train_dataloader) % 10 == 9:
                    writer.add_scalar("Loss/train", running_loss/10, i+epoch*len(train_dataloader))
                    running_loss = 0.0

            number = 0
            loss_sum = 0
            total = 0
            correct = 0
            # if epoch % 20 == 19:
            #     for tag, parm in net.named_parameters():
            #         writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

            with torch.no_grad():
                for i, data in enumerate(test_dataloader, 0):
                    inputs = data['image'].double().to(device)
                    labels = data['label'].to(device)

                    optimizer.zero_grad()
                    outputs = net(inputs).double()
                    loss = criterion(outputs, labels)
                    predicted = torch.max(outputs, 1).indices
                    number += 1
                    loss_sum += loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            writer.add_scalar("Loss/test", running_loss/number, epoch)
            writer.add_scalar("Accuracy/test", correct/total, epoch)

        if not os.path.exists('models/'+self.name):
            os.mkdir('experiments/'+self.name+'/model_weights')
        PATH = 'experiments/'+self.name+'/model_weights/model.pth'
        torch.save(net.state_dict(), PATH)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 200), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class VAETrainer:
    def __init__(self, name, network, train_dataloader, test_dataloader, optimizer, scheduler = None):
        self.name = name
        self.net = network
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.criterion = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, number_of_epochs):
        writer = SummaryWriter(log_dir='experiments/' + str(self.name))
        device = self.device
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader
        net = self.net
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        net.to(device)

        # criterion = self.criterion
        # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 250], gamma=0.1)

        for epoch in tqdm.tqdm(range(number_of_epochs)):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs = data['image'].to(device)
                labels = data['label'].to(device)

                optimizer.zero_grad()
                recon_batch, mu, logvar = net(inputs)
                loss = loss_function(recon_batch, inputs, mu, logvar)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i + epoch * len(train_dataloader) % 10 == 9:
                    writer.add_scalar("Loss/train", running_loss / 10, i + epoch * len(train_dataloader))
                    running_loss = 0.0
            with torch.no_grad():
                outputs = None
                all_labels = None
                for i, data in enumerate(train_dataloader, 0):
                    inputs = data['image'].to(device)
                    labels = data['label'].to(device)
                    optimizer.zero_grad()
                    if outputs is None:
                        logvar, mu = net.encode(inputs)
                        outputs = net.reparameterize(logvar, mu)
                        all_labels = labels
                    else:
                        logvar, mu = net.encode(inputs)
                        outputs = torch.cat(outputs, net.reparameterize(logvar, mu))
                        all_labels = torch.cat(all_labels, labels)
                model = SVC()
                model.fit(outputs.numpy(), all_labels.numpy())
                correct = 0
                total = 0
                for i, data in enumerate(test_dataloader, 0):
                    inputs = data['image'].to(device)
                    labels = data['label'].numpy()
                    optimizer.zero_grad()
                    logvar, mu = net.encode(inputs)
                    outputs = net.reparameterize(logvar, mu)
                    predicted = model.predict(outputs.numpy())
                    total += len(labels)
                    correct += np.sum(predicted == labels)
                writer.add_scalar("Accuracy/test", correct / total, epoch)

        if not os.path.exists('models/' + self.name):
            os.mkdir('experiments/' + self.name + '/model_weights')
        PATH = 'experiments/' + self.name + '/model_weights/model.pth'
        torch.save(net.state_dict(), PATH)