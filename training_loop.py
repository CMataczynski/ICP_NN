import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

import tqdm
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from torch.utils.tensorboard import SummaryWriter
from utils import plot_confusion_matrix


class Trainer:
    def __init__(self, name, network, train_dataloader, test_dataloader, criterion, optimizer, scheduler=None):
        self.name = name
        self.net = network
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, number_of_epochs):
        writer = SummaryWriter(log_dir='experiments/' + str(self.name))
        device = self.device
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader
        net = self.net.double()
        dummy_input = torch.zeros(1,180).double()
        writer.add_graph(self.net, input_to_model=dummy_input)
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        net.to(device)
        best_net = None
        max_score = 0

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
                    writer.add_scalar("Loss/train", running_loss / 10, i + epoch * len(train_dataloader))
                    running_loss = 0.0

            number = 0
            loss_sum = 0
            total = 0
            correct = 0

            with torch.no_grad():
                preds = []
                labs = []
                for i, data in enumerate(test_dataloader, 0):
                    inputs = data['image'].double().to(device)
                    labels = data['label'].to(device)

                    optimizer.zero_grad()
                    outputs = net(inputs).double()
                    loss = criterion(outputs, labels)
                    predicted = torch.max(outputs, 1).indices
                    preds += predicted.tolist()
                    labs += labels.tolist()
                    number += 1
                    loss_sum += loss.item()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            f1 = f1_score(labs, preds, average='weighted')
            if f1 > max_score:
                max_score = f1
                best_net = copy.deepcopy(net)
            writer.add_scalar("Accuracy/test", correct / total, epoch)
            writer.add_scalar("F1_score/test", f1, epoch)
        writer.add_figure(self.name + " - Confusion Matrix",
                          plot_confusion_matrix(labs, preds, ["T1", "T2", "T3", "T4", "A+E"]))
        writer.close()
        if not os.path.exists('models/' + self.name):
            os.mkdir('experiments/' + self.name + '/model_weights')
        PATH = 'experiments/' + self.name + '/model_weights/model_final.pth'
        torch.save(net, PATH)
        PATH = 'experiments/' + self.name + '/model_weights/model_best.pth'
        torch.save(best_net, PATH)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class VAETrainer:
    def __init__(self, name, network, train_dataloader, test_dataloader, optimizer, scheduler=None):
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
        net = self.net.double()
        self.net.eval()
        dummy_input = torch.zeros(1, 180).double()
        writer.add_graph(self.net, input_to_model=dummy_input)
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        net.to(device)
        best_net = None
        max_score = 0
        for epoch in tqdm.tqdm(range(number_of_epochs)):

            net.train()
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs = data['image'].double().to(device)
                labels = data['label'].to(device)

                optimizer.zero_grad()
                recon_batch, mu, logvar = net(inputs)
                # print(recon_batch, mu, logvar)
                loss = loss_function(recon_batch, inputs, mu, logvar)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i + epoch * len(train_dataloader) % 10 == 9:
                    writer.add_scalar("Loss/train", running_loss / 10, i + epoch * len(train_dataloader))
                    running_loss = 0.0

            net.eval()
            with torch.no_grad():
                outputs = None
                all_labels = None
                for i, data in enumerate(train_dataloader, 0):
                    inputs = data['image'].double().to(device)
                    labels = data['label'].to(device)
                    optimizer.zero_grad()
                    if outputs is None:
                        outputs, mu, logvar = net.encode(inputs)
                        all_labels = labels
                    else:
                        out, mu, logvar = net.encode(inputs)
                        outputs = torch.cat((outputs, out))
                        all_labels = torch.cat((all_labels, labels))
                model = SVC()
                # print(outputs.shape, all_labels.shape)
                model.fit(outputs.numpy(), all_labels.numpy())
                correct = 0
                total = 0
                labs = []
                preds = []
                for i, data in enumerate(test_dataloader, 0):
                    inputs = data['image'].double().to(device)
                    labels = data['label'].numpy()
                    optimizer.zero_grad()
                    outputs, mu, logvar = net.encode(inputs)
                    predicted = model.predict(outputs.numpy())
                    preds += predicted.tolist()
                    labs += labels.tolist()
                    total += len(labels)
                    correct += np.sum(predicted == labels)
                f1 = f1_score(labs, preds, average='weighted')
                if f1 > max_score:
                    max_score = f1
                    best_net = copy.deepcopy(net)
                writer.add_scalar("Accuracy/test", correct / total, epoch)
                writer.add_scalar("F1_score/test", f1, epoch)

        if "5cls" in self.name:
            writer.add_figure(self.name + " - Confusion Matrix",
                              plot_confusion_matrix(labs, preds, ["T1", "T2", "T3", "T4", "A+E"]))
        else:
            writer.add_figure(self.name + " - Confusion Matrix",
                              plot_confusion_matrix(labs, preds, ["T1", "T2", "T3", "T4"]))

        writer.close()
        if not os.path.exists('models/' + self.name):
            os.mkdir('experiments/' + self.name + '/model_weights')
        PATH = 'experiments/' + self.name + '/model_weights/model_final.pth'
        torch.save(net, PATH)
        PATH = 'experiments/' + self.name + '/model_weights/model_best.pth'
        torch.save(best_net, PATH)
