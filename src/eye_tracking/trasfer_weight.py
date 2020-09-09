import numpy  as np
from scipy import io
import pickle

import torch

from model import eyes_model, face_model, face_grid_model, eye_conect_model, full_connect_model

PATH = 'src/eye_tracking/checkpoint.pth.tar'

""" 
torch size : torch.Size([96, 3, 11, 11]) 
tf size: <tf.Variable 'conv2d_4/kernel:0' shape=(11, 11, 3, 96) First layer weight shape
torch size : torch.Size([256, 48, 5, 5]) 
tf size: <tf.Variable 'conv2d_5/kernel:0' shape=(5, 5, 48, 256)

'eyeModel.features.0.weight', 'eyeModel.features.0.bias', 'eyeModel.features.4.weight', 'eyeModel.features.4.bias', 
'eyeModel.features.8.weight', 'eyeModel.features.8.bias', 'eyeModel.features.10.weight', 'eyeModel.features.10.bias', 
'faceModel.conv.features.0.weight', 'faceModel.conv.features.0.bias', 'faceModel.conv.features.4.weight', 'faceModel.conv.features.4.bias', 
'faceModel.conv.features.8.weight', 'faceModel.conv.features.8.bias', 'faceModel.conv.features.10.weight', 'faceModel.conv.features.10.bias', 
'faceModel.fc.0.weight', 'faceModel.fc.0.bias', 'faceModel.fc.2.weight', 'faceModel.fc.2.bias', 'gridModel.fc.0.weight', 'gridModel.fc.0.bias', 
'gridModel.fc.2.weight', 'gridModel.fc.2.bias', 'eyesFC.0.weight', 'eyesFC.0.bias', 'fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias'
"""

model_dict = torch.load(PATH, map_location='cpu')
weights = model_dict['state_dict']

def change_axis(torch_arr):
    torch_arr = np.array(torch_arr)
    print(torch_arr.shape)
    if len(torch_arr.shape) == 4:
        torch_arr = np.moveaxis(torch_arr, [0, 1], [3, 2])
    elif len(torch_arr.shape) == 2:
        torch_arr = np.moveaxis(torch_arr, 0, 1)
    return torch_arr


def save_model(key_string, model_name, save_path):
    layer_weights = []
    for key in model_dict['state_dict'].keys():
        if key.startswith(key_string):
            layer_weights.append(change_axis(weights[key]))

    i  = 0
    for layer in model_name.layers:
        if len(layer.weights) > 1:
            set_weights = [layer_weights[i], layer_weights[i + 1]]
            print('in %d layer' %i)
            print(set_weights[0].shape, set_weights[1].shape)
            layer.set_weights(set_weights)
            i += 2

    model_name.save(save_path)

key_string_list = ['eyeModel', 'faceModel', 'gridModel', 'eyesFC', 'fc']
model_name_list = [eyes_model, face_model, face_grid_model, eye_conect_model, full_connect_model]
save_path_list = ['eyes_model.h5', 'face_model.h5', 'face_grid_model.h5', 'eye_conect_model.h5', 'full_connect_model.h5']
