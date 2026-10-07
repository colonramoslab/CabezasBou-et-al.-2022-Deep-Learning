
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
rfp_paths = config['rfp_directories']
gcamp_paths = config['gcamp_directories']

ip = config['inp_path']
op = config['out_path']

ilr = config['initial_learning_rate']
im_shape = config['image_shape']
dff = config['downsize_filter_factor']
padding = config['padding']
partition_shape = config['partition_shape']

preds = config['preds']

num_paths = len(rfp_paths)

def predict():

    model = unet2D_multi_multi2()
    model_name = unet2D_multi_multi2.__name__

    model.load_weights(ip)
    print('Model Loaded')

    test_shape = (1, im_shape[0], im_shape[1], 1)

    for i in range(num_paths):

        print('Preparing files for', str(i+1),'of',str(num_paths))
    
        rfp_path = rfp_paths[i]
        gcamp_path = gcamp_paths[i]

        print(rfp_path, gcamp_path)

        rfp_files = sorted([rfp_path + f for f in os.listdir(rfp_path) if f.endswith('.tif')])
        rfp_file_names = sorted([f for f in os.listdir(rfp_path) if f.endswith('.tif')])

        gcamp_files = sorted([gcamp_path + f for f in os.listdir(gcamp_path) if f.endswith('.tif')])
        gcamp_file_names = sorted([f for f in os.listdir(gcamp_path) if f.endswith('.tif')])

        num_files = len(rfp_files)

        print('Ready to predict for ' + str(i+1) + ' of ' + str(num_paths))

        for j in range(num_files):

            rfp_vol_path = rfp_files[j]
            gcamp_vol_path = gcamp_files[j]

            file_str = rfp_file_names[j]

            rfp_array = tiff.imread(rfp_vol_path)
            gcamp_array = tiff.imread(gcamp_vol_path)

            rfp_Y, rfp_X = rfp_array.shape 
            gcamp_Y, gcamp_X = gcamp_array.shape

            assert(rfp_Y == gcamp_Y and rfp_X == gcamp_X)

            if rfp_Y < 2000: # pad it

                rfp_array = np.pad(rfp_array, ((0, 2000-rfp_Y), (0,0)), 'reflect')
                gcamp_array = np.pad(gcamp_array, ((0, 2000-rfp_Y), (0,0)), 'reflect')

            elif rfp_Y > 2000:

                rfp_array = rfp_array[:2000, :]
                gcamp_array = gcamp_array[:2000, :]

            if rfp_X < 1000:

                rfp_array = np.pad(rfp_array, ((0, 0), (0, 1000-rfp_X)), 'reflect')
                gcamp_array = np.pad(gcamp_array, ((0, 0), (0, 1000-rfp_X)), 'reflect')

            elif rfp_X > 1000:

                rfp_array = rfp_array[:, :1000]
                gcamp_array = gcamp_array[:, :1000]


            rfp_array = np.expand_dims(rfp_array, 0)
            rfp_array = np.expand_dims(rfp_array, 3)

            gcamp_array = np.expand_dims(gcamp_array, 0)
            gcamp_array = np.expand_dims(gcamp_array, 3)

            mask = model.predict([rfp_array, gcamp_array])[0]

            mask = np.argmax(mask, axis=2)

            tiff.imsave(preds[i] + file_str, img_as_ubyte(mask))

            print('Finished ' + str(j+1) + ' of ' + str(num_files), rfp_file_names[j])
        print('Finished Predictions for folder ' + str(i + 1) + ' of ' + str(num_paths))
    print('Finished Predictions')

if __name__== '__main__':

    predict()
