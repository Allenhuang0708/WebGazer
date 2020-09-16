import os

import imp
import numpy  as np
from scipy import io
import pickle

import torch

import tensorflowjs as tfjs

from model import eyes_model_group1, eyes_model_group2, eyes_model_group3, face_model_group1, face_model_group2, face_model_group3, face_grid_model, eye_conect_model, full_connect_model

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

FLAG = False

def change_axis(torch_arr):
    global FLAG
    torch_arr = np.array(torch_arr)
    print(torch_arr.shape)
    if len(torch_arr.shape) == 4:
        if torch_arr.shape[1] == 48:
            torch_arr = np.moveaxis(torch_arr, [0, 1], [3, 2])
            FLAG = True
            print(torch_arr.shape)
            return [torch_arr[:, :, :, :128], torch_arr[:, :, :, -128:]]
        torch_arr = np.moveaxis(torch_arr, [0, 1], [3, 2])
    elif len(torch_arr.shape) == 2:
        torch_arr = np.moveaxis(torch_arr, 0, 1)
    elif FLAG:
        FLAG = False
        return [torch_arr[: 128], torch_arr[ -128:]]
    return torch_arr


def save_model(key_string, model, save_path):
    layer_weights = []
    for key in model_dict['state_dict'].keys():
        if key.startswith(key_string):
            layer_weights.append(change_axis(weights[key]))

    i  = 0
    if not isinstance(model, list):
        for layer in model.layers:
            if len(layer.weights) > 1:
                set_weights = [layer_weights[i], layer_weights[i + 1]]
                layer.set_weights(set_weights)
                i += 2
        SAVE_DIR = 'models/' + save_path
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        tfjs.converters.save_keras_model(model, SAVE_DIR)
    else:
        for j, group_model in enumerate(model):
            if j !=1:
                for layer in group_model.layers:
                    if len(layer.weights) > 1:
                        set_weights = [layer_weights[i], layer_weights[i + 1]]
                        layer.set_weights(set_weights)
                        i += 2
                SAVE_DIR = 'models/' + save_path + '_groups' + str(j)
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                tfjs.converters.save_keras_model(group_model, SAVE_DIR)
            else:
                for k in range(2):
                    for layer in group_model.layers:
                        if len(layer.weights) > 1:
                            print(layer_weights[i][k].shape)
                            set_weights = [layer_weights[i][k], layer_weights[i + 1][k]]
                            layer.set_weights(set_weights)
                            
                    SAVE_DIR = 'models/' + save_path + '_groups' + str(j) + str(k)
                    if not os.path.exists(SAVE_DIR):
                        os.makedirs(SAVE_DIR)
                    tfjs.converters.save_keras_model(group_model, SAVE_DIR)
                i += 2
eyes_model_groups = [eyes_model_group1, eyes_model_group2, eyes_model_group3]
face_model_groups = [face_model_group1, face_model_group2, face_model_group3]
key_string_list = ['eyeModel', 'faceModel', 'gridModel', 'eyesFC', 'fc']
model_name_list = [eyes_model_groups, face_model_groups, face_grid_model, eye_conect_model, full_connect_model]
save_path_list = ['eyes_model', 'face_model', 'face_grid_model', 'eye_conect_model', 'full_connect_model']

""" for key_string, model, save_path in zip(key_string_list, model_name_list, save_path_list):
    save_model(key_string, model, save_path) """

from model import dense_model
layer_weights = []
for key in weights.keys():
    if key.startswith('fc.2'):
        layer_weights.append(change_axis(weights[key]))


for layer in dense_model.layers:
    if len(layer.weights) > 1:
        set_weights = [layer_weights[0], layer_weights[1]]
        layer.set_weights(set_weights)
SAVE_DIR = 'models/' + 'dense_layer'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
tfjs.converters.save_keras_model(dense_model, SAVE_DIR)
