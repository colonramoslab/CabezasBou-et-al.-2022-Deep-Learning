
import tifffile as tiff
import os
import numpy as np
import sys
from config import config
import csv

from model import *
from util import *
from metrics import *
from skimage import img_as_ubyte

home = config['home']
raw_paths = config['raw_directories']
ip = config['inp_path']
op = config['out_path']
ilr = config['initial_learning_rate']
im_shape = config['image_shape']
dff = config['downsize_filter_factor']
padding = config['padding']
partition_shape = config['partition_shape']
preds = config['preds']
nvcs = config['nvcs']


def predict(paths=raw_paths, model_path=ip):

    model = unet2D_multi()
    model_name = unet2D_multi.__name__

    model.load_weights(model_path)
    print('Model Loaded')

    test_shape = (1, im_shape[0], im_shape[1], 1)

    for i in range(len(paths)):

        print('Preparing files for', str(i+1),'of',str(len(paths)), paths[i])
        path = paths[i]

        all_files = sorted([path + f for f in os.listdir(path) if f.endswith('.tif')])
        all_file_names = sorted([f for f in os.listdir(path) if f.endswith('.tif')])
        test_files = np.array([tiff.imread(all_files[w]) for w in range(len(all_files))])

        original_shape = test_files.shape # [n, Z, Y, X]

        print('Ready to predict for ' + str(i+1) + ' of ' + str(len(paths)), paths[i])

        for j in range(original_shape[0]):

            vol_name = all_files[j]
            file_str = all_file_names[j]

            vol = test_files[j]

            Y, X = vol.shape

            if Y < 2000: # pad it

                vol = np.pad(vol, ((0, 2000-Y), (0,0)), 'reflect')

            elif Y > 2000:

                vol = vol[48:, :]

            if X < 1000:

                vol = np.pad(vol, ((0,0), (0, 1000-X)), 'reflect')

            elif X > 1000:

                vol = vol[:, 9:]

            vol = np.expand_dims(vol, 0)
            vol = np.expand_dims(vol, 3)

            mask = model.predict(vol)[0]
            mask = np.argmax(mask, axis=2)

            tiff.imsave(preds[i] + file_str, img_as_ubyte(mask))

            print('Finished ' + str(j+1) + ' of ' + str(test_files.shape[0]), all_file_names[j])
        print('Finished Predictions for folder ' + str(i + 1) + ' of ' + str(len(paths)))
    print('Finished Predictions')

if __name__== '__main__':

    predict()
