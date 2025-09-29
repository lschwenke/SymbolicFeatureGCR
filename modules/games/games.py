"""This Module gathers game functions that can be used to test the approximation methods
Adapted from: https://github.com/mmschlk/shapiq
"""
import random
import math

import numpy as np
import torch

def _sigmoid(x):
    return 1 / (1 + math.exp(-x))
	
class DLMetaGame:

    device = "cuda" if torch.cuda.is_available() else "cpu"


    def __init__(self, model, data, modelType='Transforer', random_seed=42, regression=False):
        self.regression = regression
        self.modelType = modelType
        self.x_data = data.copy().squeeze()
        self.n_samples = len(self.x_data)
        self.replacement_values = np.mean(self.x_data, axis=0).reshape(1, -1)
        self.n = len(self.replacement_values[0])
        self.model = model
        self.empty_value = self.call_model(self.replacement_values, set())
        #print(self.model.score(self.x_data, self.y_data))

    def call_model(self, x_i: np.ndarray, S: set):
        x_input = np.zeros(shape=(1, self.n))
        x_input[:, :] = self.replacement_values[:]
        #x_input[:, :] = x_i[:]*-1
        x_input[:, tuple(S)] = x_i[:, tuple(S)]
        with torch.no_grad():
            self.model.eval()
            x_input = np.expand_dims(x_input,2)
            if self.modelType == 'CNN':
                x_input = np.expand_dims(x_input,1)
            input_ids = torch.from_numpy(x_input).to(self.device)

            preds = self.model(input_ids.double())
        
        #print('---')
        #print(x_input)
        #print(np.array(preds.cpu()))
        return np.array(preds.cpu())[0]#preds.argmax(dim=1).cpu().detatch()[0]


class DLGame:

    device = "cuda" if torch.cuda.is_available() else "cpu"


    def __init__(self, meta_game: DLMetaGame, data_index = None, set_zero: bool = True):
        if data_index is None:
            data_index = random.randint(0, meta_game.n_samples - 1)
        assert data_index <= meta_game.n_samples - 1, "Not enough data in this dataset."

        self.meta_game = meta_game
        self.data_point = meta_game.x_data[data_index]
        self.data_point = self.data_point.reshape(1, -1)
        
        with torch.no_grad():
            meta_game.model.eval()
            data_point = np.expand_dims(self.data_point.copy(),2)

            if self.meta_game.modelType == 'CNN':
                data_point = np.expand_dims(data_point,1)

            input_ids = torch.from_numpy(data_point).to(self.device) 

            preds = meta_game.model(input_ids.double()).cpu()
        self.targetIndex = np.argmax(preds)


        self.empty_value = 0
        if set_zero:
            self.empty_value = self.meta_game.empty_value

        self.n = meta_game.n
        self.game_name = "Andor"

    def set_call(self, S):
        output = self.meta_game.call_model(x_i=self.data_point, S=S)
        #print('####')
        #print(output)
        #print(self.empty_value)
        return output[self.targetIndex] - self.empty_value[self.targetIndex]