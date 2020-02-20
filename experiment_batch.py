import os

import torch
from torch import nn

from experiment_manager import Manager
from models.AEmodel import VAE, CNNVAE, AE, CNNAE
from models.CNNmodel import CNN
from models.FCmodel import FCmodel
from models.RNNmodel import LSTM, GRU, LSTMFCN
from utils import Initial_dataset_loader, get_fourier_coeff
import pandas as pd
import numpy as np

if __name__ == "__main__":
    loading = [None,
               get_fourier_coeff,
               lambda x, y: np.polynomial.chebyshev.chebfit(x, y, 7),
               lambda x, y: np.polynomial.legendre.legfit(x, y, 7),
               lambda x, y: np.polynomial.hermite_e.hermefit(x, y, 7)]
    inputs = [180, 178, 8, 8, 8]
    batch_names = ["Raw_", "Fourier_", "Cheb_7_", "Legendre_7_", "Hermite_e_7_"]
    FC_params = [(32, 16), (32, 16), (8, 4), (8, 4), (8, 4)]

    for load, input_size, batch_name, fc_param in zip(loading, inputs, batch_names, FC_params):
        result_dataframe = pd.DataFrame(columns=["Nazwa", "Parametry", "Accuracy [%]", "F1 Score"])
        names = ["FC_full", "CNN", "LSTM_full", "GRU_full", "LSTM_FCN_full", "VAE", "CNN_VAE"]
        # names = ["LSTM_FCN_full", "VAE", "CNN_VAE"]
        rootdir = os.path.join(os.getcwd(), 'experiments')
        dataset = "full_corrected_dataset"
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
        log = []
        for name in names:
            if name == "FC_full":
                model = FCmodel(input_size, 4, fc_param[0], fc_param[1])
                name_full = batch_name + name + "_4cls_weighted"
                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 4, fc_param[0], fc_param[1])
                name_full = batch_name + name + "_4cls_weighted"
                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True,
                                                weight_decay=0.0001)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 4, fc_param[0], fc_param[1])
                name_full = batch_name + name + "_4cls_weighted"
                try:

                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 4, fc_param[0], fc_param[1])
                name_full = name + "_4cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 4, fc_param[0], fc_param[1])
                name_full = name + "_4cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True,
                                                weight_decay=0.0001)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 4, fc_param[0], fc_param[1])
                name_full = name + "_4cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 5, fc_param[0], fc_param[1])
                name_full = name + "_5cls_weighted"
                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 5, fc_param[0], fc_param[1])
                name_full = name + "_5cls_weighted"
                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True,
                                                weight_decay=0.0001)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 5, fc_param[0], fc_param[1])
                name_full = name + "_5cls_weighted"
                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 5, fc_param[0], fc_param[1])
                name_full = name + "_5cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 5, fc_param[0], fc_param[1])
                name_full = name + "_5cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True,
                                                weight_decay=0.0001)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = FCmodel(input_size, 5, fc_param[0], fc_param[1])
                name_full = name + "_5cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "LSTM_full":
                model = LSTM()
                name_full = name + "_4cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM(output_size=5)
                name_full = name + "_5cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM()
                name_full = batch_name + name + "_4cls_weighted"
                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM(output_size=5)
                name_full = name + "_5cls_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM(hidden_layer_size=16)
                name_full = name + "_4cls_hidden"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM(hidden_layer_size=16, output_size=5)
                name_full = name + "_5cls_hidden"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM(hidden_layer_size=16, bidirectional=True)
                name_full = name + "_4cls_hidden_bidir_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTM(hidden_layer_size=16, output_size=5, bidirectional=True)
                name_full = name + "_5cls_hidden_bidir_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "GRU_full":
                model = GRU()
                name_full = name + "_4cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU(output_size=5)
                name_full = name + "_5cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU()
                name_full = batch_name + name + "_4cls_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU(output_size=5)
                name_full = name + "_5cls_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU(hidden_layer_size=16)
                name_full = name + "_4cls_hidden"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU(hidden_layer_size=16, output_size=5)
                name_full = name + "_5cls_hidden"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU(hidden_layer_size=16, bidirectional=True)
                name_full = name + "_4cls_hidden_bidir_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = GRU(hidden_layer_size=16, output_size=5, bidirectional=True)
                name_full = name + "_5cls_hidden_bidir_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "LSTM_FCN_full":
                model = LSTMFCN(input_size, 4)
                name_full = name + "_4cls"

                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue
                model = LSTMFCN(input_size, 6)
                name_full = name + "_5cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTMFCN(input_size, 4)
                name_full = batch_name + name + "_4cls_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = LSTMFCN(input_size, 5)
                name_full = name + "_5cls_weighted"


                try:
                    criterion = nn.CrossEntropyLoss(weights_5cls)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "VAE":
                model = VAE(sizes=[input_size, max(input_size//4,6), max(input_size//8,4)])
                name_full = name + "_4cls"


                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, None, optimizer, VAE=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = VAE(sizes=[input_size, max(input_size//4,6), max(input_size//8,4)])
                name_full = name + "_5cls"

                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, None, optimizer, VAE=True, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "CNN_VAE":
                model = CNNVAE(full_features=input_size, squeeze_size=max(input_size//8, 4))
                name_full = name + "_4cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, None, optimizer, VAE=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = CNNVAE(full_features=input_size, squeeze_size=max(input_size//8, 4))
                name_full = name + "_5cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, None, optimizer, VAE=True, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "AE":
                model = AE(sizes=[input_size, max(input_size//4, 6), max(input_size//8,4)])
                name_full = name + "_4cls"


                try:
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = AE(sizes=[input_size, max(input_size//4,6), max(input_size//8,4)])
                name_full = name + "_5cls"

                try:
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=True, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "CNNAE":
                model = CNNAE(full_features=input_size, squeeze_size=max(input_size//8, 4))
                name_full = name + "_4cls"
                try:
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = CNNAE(full_features=input_size, squeeze_size=max(input_size//8, 4))
                name_full = name + "_5cls"
                try:
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=True, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

            if name == "CNN":
                model = CNN(input_size, 4)
                name_full = name + "_4cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = CNN(input_size, 5)
                name_full = name + "_5cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = CNN(input_size, 4)
                name_full = name + "_4cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
                    continue

                model = CNN(input_size, 5)
                name_full = name + "_5cls"
                try:
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    manager = Manager(name_full, model, dataset, criterion, optimizer, VAE=False, full=True)
                    manager.run(1000)
                    result_dataframe.append(manager.get_results())
                except:
                    log.append("failed model " + name_full)
        result_dataframe.to_csv(os.path.join(os.getcwd(), "results", batch_name+".csv"))
        print(log)
