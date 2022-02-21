import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
import time
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from utils import plot_confusion_matrix, inf_generator, RunningAverageMeter
from models.Decoders import FCDecoder

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class Trainer:
    def __init__(self, name, network, train_dataloader, test_dataloader, criterion,
                optimizer, class_dict, metrics, output_to_pred_fcn, loss_preprocessing=None,
                leading_metric=None, scheduler=None, nfe_logging = False,
                input_preprocessing=None):
        self.name = name
        #string with exp name
        self.net = network
        #torch network

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        #dataloaders

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.criterion = criterion
        #criterion function with arguments: outputs, labels
        self.optimizer = optimizer
        #torch optimizer as a lambda with parameters tbd
        self.scheduler = scheduler
        #scheduler as a learning_rate_fn
        self.metrics = metrics
        #dictionary of metrics fcn {name: fcn(true, pred)}
        if leading_metric is not None:
            self.leading_metric = leading_metric
        else:
            self.leading_metric = list(metrics.keys())[0]
        #str - name of leading metric from metrics
        self.nfe_logging = nfe_logging

        self.output_to_pred_fcn = output_to_pred_fcn

        if loss_preprocessing is not None:
            self.loss_preprocessing = loss_preprocessing
        else:
            self.loss_preprocessing = lambda x: x

        self.class_dict = class_dict

        if input_preprocessing is not None:
            self.input_preprocessing = input_preprocessing
        else:
            self.input_preprocessing = lambda x: x

    def train(self, number_of_epochs):
        model = self.net
        device = self.device
        name = self.name
        writer = SummaryWriter(log_dir='experiments/' + str(name))
        makedirs(os.path.join(os.getcwd(), "experiments", name))

        model = model.float().to(device)
        criterion = self.criterion

        train_loader, test_loader = self.train_dataloader, self.test_dataloader
        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)

        lr_fn = self.scheduler

        optimizer = self.optimizer(model.parameters())

        best_leading_metric = 0.0
        batch_time_meter = RunningAverageMeter()
        loss_meter = RunningAverageMeter()
        if self.nfe_logging:
            f_nfe_meter = RunningAverageMeter()
            b_nfe_meter = RunningAverageMeter()
        end = time.time()

        for itr in tqdm(range(number_of_epochs * batches_per_epoch)):
            if lr_fn is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_fn(itr)

            model.train()
            optimizer.zero_grad()
            dct = data_gen.__next__()
            # print(dct)
            model_input = [dct[key].float().to(device) for key in dct.keys() if "data" in key]
            y = dct["label"].to(device)
            # x = x.unsqueeze(1)
            model_input = self.input_preprocessing(model_input)
            logits = model(*model_input)
            logits = self.loss_preprocessing(logits)
            loss = criterion(logits, y)

            if self.nfe_logging:
                nfe_forward = model.feature_layers[0].nfe
                model.feature_layers[0].nfe = 0

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            if itr % 10 == 9:
                writer.add_scalar("Loss/train", loss_meter.val, itr)
            if self.nfe_logging:
                nfe_backward = model.feature_layers[0].nfe
                model.feature_layers[0].nfe = 0
                f_nfe_meter.update(nfe_forward)
                b_nfe_meter.update(nfe_backward)

            batch_time_meter.update(time.time() - end)
            end = time.time()

            if itr % batches_per_epoch == 0:
                with torch.no_grad():
                    model.eval()
                    preds = []
                    labs = []
                    for i, data in enumerate(test_loader, 0):
                        model_input = [data[key].float().to(device) for key in data.keys() if "data" in key]
                        labels = data['label'].to(device)
                        model_input = self.input_preprocessing(model_input)
                        outputs = model(*model_input)
                        predicted = self.output_to_pred_fcn(outputs)
                        # if len(predicted.tolist()) != len(labels.tolist()):
                        #     print(len(model_input[0].tolist()))
                        #     print(len(labels.tolist()))
                        preds += predicted.tolist()
                        labs += labels.tolist()
                    for metric in self.metrics.keys():
                        metric_val = self.metrics[metric](labs, preds)
                        if metric == self.leading_metric and metric_val > best_leading_metric:
                            best_leading_metric = metric_val
                            torch.save({'state_dict': model.state_dict()},
                                        os.path.join(os.getcwd(), "experiments",
                                                    name, 'model_best.pth'))
                        writer.add_scalar(metric + "/test", metric_val, itr // batches_per_epoch)
                    if self.nfe_logging:
                        writer.add_scalar("NFE-F", f_nfe_meter.val, itr // batches_per_epoch)
                        writer.add_scalar("NFE-B", b_nfe_meter.val, itr // batches_per_epoch)

        labs = []
        preds = []
        for data in test_loader:
            with torch.no_grad():
                model.eval()
                model_input = [data[key].float().to(device) for key in data.keys() if "data" in key]
                labels = data['label'].to(device)
                model_input = self.input_preprocessing(model_input)
                outputs = model(*model_input)
                predicted = self.output_to_pred_fcn(outputs)
                preds += predicted.tolist()
                labs += labels.tolist()

        torch.save({'state_dict': model.state_dict()},
                    os.path.join(os.getcwd(), "experiments",
                            name, 'model_last.pth'))
        try:
            labs = [self.class_dict[a] for a in labs]
            preds = [self.class_dict[a] for a in preds]
            other_metrics = [[metric, [self.metrics[metric](labs, preds)]] for metric in self.metrics if metric is not self.leading_metric]    
            writer.add_figure(name + " - Confusion Matrix",
                            plot_confusion_matrix(labs, preds,
                          [self.class_dict[key] for key in self.class_dict.keys()]))
        except:
            other_metrics = []
            pass 
        writer.close()
        return [self.leading_metric, [best_leading_metric]], other_metrics

class PreTrainer:
    def __init__(self, name, network, train_dataloader, criterion,
                optimizer, loss_preprocessing=None, scheduler=None):
        self.name = name
        #string with exp name
        self.encoder_net = network
        #torch network
        self.decoder_net = FCDecoder(network.embed_size(), 2 * 180)

        self.train_dataloader = train_dataloader
        #dataloaders

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.criterion = criterion
        #criterion function with arguments: outputs, labels
        self.optimizer = optimizer
        #torch optimizer as a lambda with parameters tbd
        self.scheduler = scheduler

        if loss_preprocessing is not None:
            self.loss_preprocessing = loss_preprocessing
        else:
            self.loss_preprocessing = lambda x: x

    def train(self, number_of_epochs):
        model = self.encoder_net
        device = self.device
        name = self.name
        decoder = self.decoder_net
        writer = SummaryWriter(log_dir='experiments/' + str(name))
        makedirs(os.path.join(os.getcwd(), "experiments", name))

        model = model.float().to(device)
        decoder = decoder.float().to(device)
        criterion = self.criterion

        train_loader = self.train_dataloader
        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)

        lr_fn = self.scheduler
        params = list(model.parameters()) + list(decoder.parameters())
        optimizer = self.optimizer(params)

        batch_time_meter = RunningAverageMeter()
        loss_meter = RunningAverageMeter()
        end = time.time()

        for itr in tqdm(range(number_of_epochs * batches_per_epoch)):
            # print("start: {}".format(time.time() - end))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(itr)
            # print("params_set: {}".format(time.time() - end))
            model.train()
            decoder.train()
            optimizer.zero_grad()
            # print("models_working: {}".format(time.time() - end))
            dct = data_gen.__next__()
            model_input = [torch.reshape(dct[key].float(), (-1, 180)).to(device) for key in dct.keys() if "data" in key]
            # print("data_generated: {}".format(time.time() - end))
            embedding = model(*model_input)
            logits = decoder(embedding)
            logits = self.loss_preprocessing(logits) * 100
            model_input = torch.stack(model_input, dim=-1) * 100
            # print("predictions_made: {}".format(time.time() - end))
            loss = criterion(logits, model_input)
            # print("loss_counted: {}".format(time.time() - end))
            loss.backward()
            optimizer.step()
            # print("optimizer_step: {}".format(time.time() - end))
            loss_meter.update(loss.item())
            if itr % 10 == 0:
                writer.add_scalar("Loss/train", loss_meter.val, itr)
                torch.save({'state_dict': model.state_dict()},
                            os.path.join(os.getcwd(), "experiments",
                                        name, 'model_final.pth'))
            batch_time_meter.update(time.time() - end)
            writer.add_scalar("batch_time/train", batch_time_meter.val, itr)
            end = time.time()

        writer.close()
        return ["Loss", loss_meter.val], [None]
