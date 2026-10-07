
import tifffile as tiff
import os
import numpy as np
import sys
from config import config
import csv

from model import *
from util import pad_vols
from metrics import *
from skimage import img_as_ubyte

home = config['home']
rfp_paths = config['raw_directories']
gcamp_paths = config['GCaMP_directories']
ip = config['inp_path']
preds = config['preds']
masks = config['masks']
im_shape = config['image_shape']
threshold = config['threshold']
data_path = config['op']


def predict_val():

    # model = unet2D_multi()
    # model.load_weights(ip)
    # print('Model Loaded')

    model = unet2D_multi_multi2()
    model.load_weights(ip)
    print('Model Loaded')

    test_shape = (1, im_shape[0], im_shape[1], 1)

    all_RFP_files = []
    all_GCaMP_files = []

    all_file_names = []

    num_paths = len(rfp_paths)

    for i in range(num_paths):
        
        RFP_files = sorted([rfp_paths[i] + f for f in os.listdir(rfp_paths[i]) if f.endswith('.tif')])
        GCaMP_files = sorted([gcamp_paths[i] + f for f in os.listdir(gcamp_paths[i]) if f.endswith('.tif')])

        files = sorted([f for f in os.listdir(rfp_paths[i]) if f.endswith('.tif')])
    
        all_RFP_files.extend(RFP_files)
        all_GCaMP_files.extend(GCaMP_files)

        all_file_names.extend(files)

    n = len(all_RFP_files)

    train_locs = np.load(data_path + '/train_locs.npy')
    num_train = len(train_locs)
    num_val = int(np.floor(num_train*pct_val))
    val_locs = train_locs[-num_val:]

    print(n, val_locs)

    for j in val_locs:

        rfp_path = all_RFP_files[j]
        gcamp_path = all_GCaMP_files[j]
        vol_name = all_file_names[j]

        pred_path = rfp_path.split('raw_crop')[0]

        rfp = tiff.imread(rfp_path)
        rfp = rfp[:2000, :1000]
        rfp = np.expand_dims(rfp, 0)
        rfp = np.expand_dims(rfp, 3)

        gcamp = tiff.imread(gcamp_path)
        gcamp = gcamp[:2000, :1000]
        gcamp = np.expand_dims(gcamp, 0)
        gcamp = np.expand_dims(gcamp, 3)

        mask = model.predict([rfp, gcamp])[0]
        mask = np.argmax(mask, axis=2)

        tiff.imsave(pred_path + 'Predictions/' + vol_name, img_as_ubyte(mask))

        print('Finished', j, all_file_names[j])
        
    print('Finished Predictions')

if __name__== '__main__':

    predict_val()
