import argparse
import os
from torch import nn
import torch.optim
from utils import Initial_dataset_loader
from torch.utils.data import DataLoader
from training_loop import Trainer, VAETrainer
from models.FCmodel import FCmodel
from models.RNNmodel import LSTM, GRU, LSTMFCN
from models.AEmodel import VAE
from experiment_manager import Manager
import numpy as np

if __name__ == "__main__":
    names = ["FC_full", "LSTM_full", "GRU_full", "LSTM_FCN_full", "VAE"]
    rootdir = os.path.join(os.getcwd(), 'experiments')

    dataset = "full_dataset"
    datasets = os.path.join(os.getcwd(), "datasets", dataset)
    train_dataset_path = os.path.join(datasets, "train")
    train_dataset = Initial_dataset_loader(train_dataset_path)
    weights = train_dataset.get_class_weights()
    train_dataset = Initial_dataset_loader(train_dataset_path, full=True)
    weights_6cls = train_dataset.get_class_weights()
    train_dataset = None
    log = []
    for name in names:
        max = 0
        for subdir in os.listdir(rootdir):
            if name in subdir:
                var = int(subdir.split('_')[-1])
                if var > max:
                    max = var
        name = name

        if name == "FC_full":
            model = FCmodel(180, 4)
            name_full = name + "_4cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            criterion = nn.CrossEntropyLoss(weights)

            try:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1


            model = FCmodel(180, 4)
            name_full = name + "_4cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 4)
            name_full = name + "_4cls_weighted"
            name_full = name_full + "_" + str(max+1)
            try:
                dataset = "full_dataset"
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 4)
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 4)
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 4)
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 6)
            name_full = name + "_6cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 6)
            name_full = name + "_6cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 6)
            name_full = name + "_6cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 6)
            name_full = name + "_6cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 6)
            name_full = name + "_6cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = FCmodel(180, 6)
            name_full = name + "_6cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

        if name == "LSTM_full":
            model = LSTM()
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM(output_size=6)
            name_full = name + "_6cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM()
            name_full = name + "_4cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM(output_size=6)
            name_full = name + "_6cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM(hidden_layer_size=16)
            name_full = name + "_4cls_hidden"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM(hidden_layer_size=16, output_size=6)
            name_full = name + "_6cls_hidden"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM(hidden_layer_size=16,bidirectional=True)
            name_full = name + "_4cls_hidden_bidir_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTM(hidden_layer_size=16, output_size=6, bidirectional=True)
            name_full = name + "_6cls_hidden_bidir_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

        if name == "GRU_full":
            model = GRU()
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU(output_size=6)
            name_full = name + "_6cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU()
            name_full = name + "_4cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU(output_size=6)
            name_full = name + "_6cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU(hidden_layer_size=16)
            name_full = name + "_4cls_hidden"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU(hidden_layer_size=16, output_size=6)
            name_full = name + "_6cls_hidden"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU(hidden_layer_size=16, bidirectional=True)
            name_full = name + "_4cls_hidden_bidir_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = GRU(hidden_layer_size=16, output_size=6, bidirectional=True)
            name_full = name + "_6cls_hidden_bidir_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

        if name == "LSTM_FCN_full":
            model = LSTMFCN(180,4)
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTMFCN(180,6)
            name_full = name + "_6cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTMFCN(180,4)
            name_full = name + "_4cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

            model = LSTMFCN(180,6)
            name_full = name + "_6cls_weighted"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss(weights_6cls)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

        if name == "VAE":
            model = VAE()
            name_full = name + "_4cls"
            name_full = name_full + "_" + str(max+1)
            dataset = "full_dataset"
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=True)
                manager.run(1000)
            except:
                log.append("failed model " + name_full)
            max = max+1

    print(log)