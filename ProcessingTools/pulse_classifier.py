import gc
import numpy as np
import torch
import scipy
from scipy import interpolate
from typing import List

from ProcessingTools.model.ResnetModel import ResNet

class Classifier:
    def __init__(self, params: dict) -> None:
        self.model_weights = params["classification"]["model_weights"]
        self.batch_size = params["classification"]["batch_size"]
        self.gpu = params["classification"]["gpu"]
        self.resampling = params["classification"]["resampling"]

        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() and self.gpu else 'cpu')
        self.model = ResNet(5).to(self.device)
        self.model.load_state_dict(torch.load(self.model_weights, map_location=self.device)["state_dict"])
        self.model.eval()

    
    def preprocess(self, pulse):
        data = pulse
        interp = interpolate.interp1d(np.arange(0, len(data), 1), data,
                                    kind="cubic")
        new_t = np.linspace(0, len(data)-1, self.resampling)
        data = interp(new_t)
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)

        return data

    def classify_batch(self, pulse_inp: List[np.ndarray]) -> List:
        predictions = []
        pulses = []
        for idp in range(0, len(pulse_inp)):
            data = self.preprocess(pulse_inp[idp])
            pulses.append(data)
            if len(pulses) % self.batch_size == self.batch_size - 1:
                tensors = torch.tensor(pulses, dtype=torch.float).to(self.device)
                tensors = tensors.unsqueeze(1)
                outputs = self.model(tensors).detach().cpu().tolist()
                
                del tensors
                torch.cuda.empty_cache()
                gc.collect()

                predictions += outputs
                pulses = []
        if len(pulses) > 0:
            tensors = torch.tensor(pulses, dtype=torch.float).to(self.device)
            tensors = tensors.unsqueeze(1)
            outputs = self.model(tensors).detach().cpu().tolist()
            
            del tensors
            torch.cuda.empty_cache()
            gc.collect()

            predictions += outputs
            pulses = []
            
        return predictions