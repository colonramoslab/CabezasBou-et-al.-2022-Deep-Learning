
import random
import os
import numpy as np
from keras.losses import binary_crossentropy
from config import config
import glob as glob
import tifffile as tiff
import sys
import time
from skimage import img_as_ubyte, img_as_int
from scipy import ndimage
from keras.utils import to_categorical

from metrics import *

padding = config['padding']
pad_each_side = int(padding/4)

#########################################

home = config['home']
rfp_paths = config['rfp_directories']
gcamp_paths = config['gcamp_directories']

ip = config['inp_path']

ilr = config['initial_learning_rate']
im_shape = config['image_shape']
dff = config['downsize_filter_factor']

padding = config['padding']
partition_shape = config['partition_shape']

#########################################

def get_chunk(volume, x_start, x_end, y_start, y_end, z_start, z_end):
    
    chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
    
    return chunk

def partition2(shaped, new_shape):

    # n is the number of large volumes in the set
    n = shaped.shape[0]

    # shaped is the array of data to be partitioned
    # array of [n, z1, y1, x1]
    z1 = shaped.shape[1]
    y1 = shaped.shape[2]
    x1 = shaped.shape[3]

    # new_shape is a 3-tuple of the form (z,y,x)
    new_z = new_shape[0]
    new_y = new_shape[1]
    new_x = new_shape[2]

    # The factors of growth for each dimension:
    z_rat = z1 / new_z
    y_rat = y1 / new_y
    x_rat = x1 / new_x

    # number of total volumes
    n_large = int(z_rat * y_rat * x_rat * n)

    partitions = np.empty((n_large, new_z, new_y, new_x))
    factor = int(n_large/n)
    
    for j in range(n):
        
        i = 0
        Z = 0
        chunk = np.empty((factor, new_z, new_y, new_x))
        
        while Z < z_rat:
            Y = 0
            while Y < y_rat:
                X = 0
                while X < x_rat:
                    chunk[i] = shaped[j, Z * new_z:(Z + 1) * new_z, Y * new_y:(Y + 1) * new_y,
                               X * new_x:(X + 1) * new_x]
                    i += 1
                    X += 1
                Y += 1
            Z += 1

        #partitions = np.concatenate([partitions, chunk], axis=0)
        partitions[j*factor:(j+1)*factor] = chunk
        
    return (partitions)

def split_into_eight(cropped_volumes, padding, data_type):

    print('Starting to partition')

    t1 = time.time()

    # n is the number of large volumes in the set
    n = cropped_volumes.shape[0]

    Z_crop = cropped_volumes.shape[1] # 212 
    Y_crop = cropped_volumes.shape[2] # 348 
    X_crop = cropped_volumes.shape[3] # 212 

    # First pad up
    cropped_volumes = np.pad(cropped_volumes, ((0,0), (padding, padding), (padding, padding), (padding, padding),(0,0)), mode='reflect')

    Z_crop_pad = Z_crop + 2*padding # 212 --> 220
    Y_crop_pad = Y_crop + 2*padding # 348 --> 356
    X_crop_pad = X_crop + 2*padding # 212 --> 220

    # Now find each partition size. It is half the total pixel count in each dimension. 
    Z_partition = int(Z_crop/2) # (212/2) = 106 
    Y_partition = int(Y_crop/2) # (348/2) = 174 
    X_partition = int(X_crop/2) # (212/2) = 106 

    # Now account for the loss of size on the other sides:
    Z_partition_pad = Z_partition + 2*padding # 106 + 4 on each side --> 114
    Y_partition_pad = Y_partition + 2*padding # 174 + 4 on each side --> 182
    X_partition_pad = X_partition + 2*padding # 106 + 4 on each side --> 114


    # number of total volumes
    factor = 8
    n_partitions = int(factor*n)

    partitions = np.empty((n_partitions, Z_partition_pad, Y_partition_pad, X_partition_pad, 1), dtype=data_type)
    
    for j in range(n):
        
        large_vol = cropped_volumes[j]

        chunk_000 = large_vol[:Z_partition_pad, :Y_partition_pad, :X_partition_pad].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1))
        chunk_001 = large_vol[:Z_partition_pad, :Y_partition_pad, X_partition:].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1))
        chunk_010 = large_vol[:Z_partition_pad, Y_partition:, :X_partition_pad].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1))
        chunk_011 = large_vol[:Z_partition_pad, Y_partition:, X_partition:].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1)) 

        chunk_100 = large_vol[Z_partition:, :Y_partition_pad, :X_partition_pad].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1)) 
        chunk_101 = large_vol[Z_partition:, :Y_partition_pad, X_partition:].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1))
        chunk_110 = large_vol[Z_partition:, Y_partition:, :X_partition_pad].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1))  
        chunk_111 = large_vol[Z_partition:, Y_partition:, X_partition:].reshape((1, Z_partition_pad, Y_partition_pad, X_partition_pad, 1)) 

        partitions[j*factor:(j+1)*factor] = np.concatenate([chunk_000, chunk_001, chunk_010, chunk_011, chunk_100, chunk_101, chunk_110, chunk_111])
    
    print('Went from',n,'to',8*n,'volumes in',time.time()-t1,'seconds')
        
    return (partitions)

def stitch(partitions, original_shape, new_shape):

    # Partitons are of shape: [n_new, z_new, y_new, x_new]
    # partition function takes in an input of [n_old, z_old, y_old, x_old] and returns the new shape.
    # This function aims to undo the partition effect after prediction.

    os_z = original_shape[0]  # 360
    os_y = original_shape[1]  # 432
    os_x = original_shape[2]  # 336

    new_z = new_shape[0]  # 72
    new_y = new_shape[1]  # 72
    new_x = new_shape[2]  # 112

    n_new = partitions.shape[0]  # 1440
    z_rat = int(os_z / new_z)  # 5
    y_rat = int(os_y / new_y)  # 6
    x_rat = int(os_x / new_x)  # 3

    os_n = int(n_new / (z_rat * y_rat * x_rat))  # 16
    reduction = int(n_new / os_n)  # 1440/16 = 90
    stitched = np.empty((os_n, os_z, os_y, os_x, 1))  # [16, 360, 432, 336]

    for j in range(os_n):

        partitions_j = partitions[j * reduction:(j + 1) * reduction]
        i = reduction
        Z = z_rat
        while Z > 0:
            Y = y_rat
            while Y > 0:
                X = x_rat
                while X > 0:

                    stitched[j, (Z - 1) * new_z:Z * new_z, (Y - 1) * new_y:Y * new_y, new_x * (X - 1):X * new_x] = partitions[i - 1]
                    i -= 1
                    X -= 1
                Y -= 1
            Z -= 1

    return stitched

def augment(raw_arrays, label_arrays):

    t1 = time.time()
    n, Y, X = raw_arrays.shape

    augmented_raws = np.empty((4*n, Y, X), dtype='int16')
    augmented_labels = np.empty((4*n, Y, X, 3), dtype='int8')

    for i in range(raw_arrays.shape[0]):

        r1 = np.expand_dims(raw_arrays[i], 0)
        l1 = np.expand_dims(label_arrays[i], 0)

        # Just Y
        r2 = np.flip(r1, axis=1)
        l2 = np.flip(l1, axis=1)

        # Just X
        r3 = np.flip(r1, axis=2)
        l3 = np.flip(l1, axis=2)

        # X and Y
        r7 = np.flip(r3, axis=1)
        l7 = np.flip(l3, axis=1)

        temp = np.vstack([r1, r2, r3, r7])
        temp2 = np.vstack([l1, l2, l3, l7])

        augmented_raws[i*4:4*(i+1)] = temp
        augmented_labels[i*4:4*(i+1)] = temp2

    t2 = time.time()

    print('We went from ' + str(n) + ' to ' + str(augmented_raws.shape[0]) + ' volumes! in ' + str(t2 - t1) + ' seconds!')
    return augmented_raws, augmented_labels

def augment_less(raw_arrays, label_arrays):

    t1 = time.time()
    s = raw_arrays.shape
    augmented_raws = np.empty((8*s[0], s[1], s[2], s[3]), dtype='int16')
    augmented_labels = np.empty((8*s[0], s[1], s[2], s[3]), dtype='int8')

    for i in range(raw_arrays.shape[0]):

        r1 = raw_arrays[i].reshape((1, s[1], s[2], s[3]))
        l1 = label_arrays[i].reshape((1, s[1], s[2], s[3]))

        # Just Z
        r2 = np.flip(r1, axis=1)
        l2 = np.flip(l1, axis=1)

        temp = np.vstack([r1, r2])
        temp2 = np.vstack([l1, l2])

        augmented_raws[i*2:2*(i+1)] = temp
        augmented_labels[i*2:2*(i+1)] = temp2

    t2 = time.time()

    print('We went from ' + str(s[0]) + ' to ' + str(augmented_raws.shape[0]) + ' volumes! in ' + str(t2 - t1) + ' seconds!')
    return augmented_raws, augmented_labels

def check_matching_volumes(path_files, path_labels):

    try:

        a = 'Decon'+path_files[0].split('Decon')[0][:-4]
        which_str = 'Decon'

    except:

        which_str = 'SPIMB'

    print(which_str)

    for j in range(len(path_files)):
                
        a = which_str + path_files[j].split(which_str)[1][:-4]
        b = which_str + path_labels[j].split(which_str)[1][:-4]

        if a != b:

            print(path_files[j], path_labels[j])
            break

def pad_vol(vol, data_type):

    # data is an array of volumes

    # shaped is the size of the array U-Net will accept.
    # It is (Number of volumes, Z, Y, X, Number of channels) ---> (Number of volumes, 72, 72, 112, 1)

    Z1 = im_shape[0] # Desired Z shape
    Y1 = im_shape[1] # Desired Y shape
    X1 = im_shape[2] # Desired X shape

    # Set the X, Y, and Z
    X = vol.shape[2]
    Y = vol.shape[1]
    Z = vol.shape[0]

    # Now set the pad width
    padding = ((0, Z1 - Z), (0, Y1 - Y), (0, X1 - X))

    # Pad the arrays

    try:

        new_vol = np.pad(vol, padding, mode='reflect')

    except:
        
        print('volume shape:', vol.shape)
        print('image shape:', im_shape)

    #return new_vol.astype(data_type)
    return new_vol

def get_files(aug, data_type='int8'):
    
    all_files = []
    all_labels = []

    for i in range(len(raw_paths)):
        
        path_files = sorted([raw_paths[i] + f for f in os.listdir(raw_paths[i]) if f.endswith('.tif')])
        path_labels = sorted([label_paths[i] + f for f in os.listdir(label_paths[i]) if f.endswith('.tif')])
        print(raw_paths[i])
                    
        check_matching_volumes(path_files, path_labels)
 
        all_files.extend(path_files)
        all_labels.extend(path_labels)

    n = len(all_files)

    raw_arrays = np.empty((n, im_shape[0], im_shape[1], im_shape[2]), dtype='int16')
    label_arrays = np.empty((n, im_shape[0], im_shape[1], im_shape[2]), dtype=data_type)

    for k in range(n):

        ra = tiff.imread(all_files[k])
        la = tiff.imread(all_labels[k])
        print(la.max(), la[la>0].min(), la.mean())
        
        raw_arrays[k] = pad_vol(ra, 'int16')
        label_arrays[k] = pad_vol(la, data_type)

    if aug == 'True':

        print('Augmenting Data!')
        raw_arrays, label_arrays = augment(raw_arrays, label_arrays)

    elif aug != 'True':

        print('No Augmentation!')

    # Find out how many observations we have:
    n_augmented = raw_arrays.shape[0]

    raw_arrays = np.expand_dims(raw_arrays, 4)
    label_arrays = np.expand_dims(label_arrays, 4)

    # Pick the train / test volumes:
    train_locs = random.sample(range(n_augmented), int(n_augmented*pct_train))
    test_locs = np.setdiff1d(range(n_augmented), train_locs)

    print('There are ' + str(len(train_locs)) + ' training volumes and ' + str(len(test_locs)) + ' test volumes.')

    print('Saving data to .npy')
    train_locs_array = np.array(train_locs)
    np.save('train_locs.npy', train_locs_array)

    train_arrays = np.take(a=raw_arrays, axis=0, indices=train_locs)
    train_label_arrays = np.take(a=label_arrays, axis=0, indices=train_locs)
    test_arrays = np.take(a=raw_arrays, axis=0, indices=test_locs)
    test_label_arrays = np.take(a=label_arrays, axis=0, indices=test_locs)

    return train_arrays, train_label_arrays, test_arrays, test_label_arrays

def load_data(load='T'):

    if load == 'RFP_GCaMP_Multi':

        all_RFP_files = []
        all_GCaMP_files = []
        all_labels = []

        num_paths = len(rfp_paths)

        for i in range(num_paths):
            
            RFP_files = sorted([rfp_paths[i] + f for f in os.listdir(rfp_paths[i]) if f.endswith('.tif')])
            GCaMP_files = sorted([gcamp_paths[i] + f for f in os.listdir(gcamp_paths[i]) if f.endswith('.tif')])
            path_labels = sorted([label_paths[i] + f for f in os.listdir(label_paths[i]) if f.endswith('.tif')])
        
            all_RFP_files.extend(RFP_files)
            all_GCaMP_files.extend(GCaMP_files)
            all_labels.extend(path_labels)

        n = len(all_RFP_files)

        raw_arrays = np.empty((n, im_shape[0], im_shape[1]), dtype='int16')
        gcamp_arrays = np.empty((n, im_shape[0], im_shape[1]), dtype='int16')
        label_arrays = np.empty((n, im_shape[0], im_shape[1], 3), dtype='int8')

        for k in range(n):

            print(all_RFP_files[k].split('raw_crop/')[1], all_GCaMP_files[k].split('GCaMP/')[1], all_labels[k].split('Multi-Binary/')[1])

            rfp = tiff.imread(all_RFP_files[k])
            gcamp = tiff.imread(all_GCaMP_files[k])
            la = tiff.imread(all_labels[k])

            la = to_categorical(la)
            
            raw_arrays[k] = rfp[:im_shape[0], :im_shape[1]]
            gcamp_arrays[k] = gcamp[:im_shape[0], :im_shape[1]]
            label_arrays[k] = la[:im_shape[0], :im_shape[1]]

        # print('Augmenting Data!')
        # raw_arrays, label_arrays = augment(raw_arrays, label_arrays)

        # if aug == 'True':

        #     print('Augmenting Data!')
        #     raw_arrays, label_arrays = augment(raw_arrays, label_arrays)

        # elif aug != 'True':

        #     print('No Augmentation!')

        # Find out how many observations we have:
        n_augmented = raw_arrays.shape[0]

        raw_arrays = np.expand_dims(raw_arrays, 3)
        gcamp_arrays = np.expand_dims(gcamp_arrays, 3)
        #label_arrays = np.expand_dims(label_arrays, 3)

        # Pick the train / test volumes:
        train_locs = random.sample(range(n_augmented), int(n_augmented*pct_train))
        test_locs = np.setdiff1d(range(n_augmented), train_locs)

        print('There are ' + str(len(train_locs)) + ' training volumes and ' + str(len(test_locs)) + ' test volumes.')

        print('Saving data to .npy')
        train_locs_array = np.array(train_locs)
        np.save('train_locs.npy', train_locs_array)

        rfp_train_arrays = np.take(a=raw_arrays, axis=0, indices=train_locs)
        gcamp_train_arrays = np.take(a=gcamp_arrays, axis=0, indices=train_locs)
        train_label_arrays = np.take(a=label_arrays, axis=0, indices=train_locs)

        rfp_test_arrays = np.take(a=raw_arrays, axis=0, indices=test_locs)
        gcamp_test_arrays = np.take(a=gcamp_arrays, axis=0, indices=test_locs)
        test_label_arrays = np.take(a=label_arrays, axis=0, indices=test_locs)

        np.save(log_path + 'rfp_train_arrays.npy', rfp_train_arrays)
        np.save(log_path + 'gcamp_train_arrays.npy', gcamp_train_arrays)
        np.save(log_path + 'train_label_arrays.npy', train_label_arrays)

        np.save(log_path + 'rfp_test_arrays.npy', rfp_test_arrays)
        np.save(log_path + 'gcamp_test_arrays.npy', gcamp_test_arrays)
        np.save(log_path + 'test_label_arrays.npy', test_label_arrays)

    elif load == 'RFP_GCaMP_Multi_Load':

        rfp_train_arrays = np.load(log_path + 'rfp_train_arrays.npy')
        gcamp_train_arrays = np.load(log_path + 'gcamp_train_arrays.npy')
        train_label_arrays = np.load(log_path + 'train_label_arrays.npy')

        rfp_test_arrays = np.load(log_path + 'rfp_test_arrays.npy')
        gcamp_test_arrays = np.load(log_path + 'gcamp_test_arrays.npy')
        test_label_arrays = np.load(log_path + 'test_label_arrays.npy')   


    else:

        print('Pick a valid load setting!')

    print('Loaded data')
    return rfp_train_arrays, gcamp_train_arrays, train_label_arrays, rfp_test_arrays, gcamp_test_arrays, test_label_arrays

def pad_vols(data, data_type):

    # data is an array of volumes

    # shaped is the size of the array U-Net will accept.
    # It is (Number of volumes, Z, Y, X, Number of channels) ---> (Number of volumes, 72, 72, 112, 1)

    Z1 = im_shape[0] # Desired Z shape
    Y1 = im_shape[1] # Desired Y shape
    X1 = im_shape[2] # Desired X shape

    shaped = np.empty((data.shape[0], im_shape[0], im_shape[1], im_shape[2], im_shape[3]), dtype=dtype)

    for i in range(len(data)):

        vol = data[i]

        # Set the X, Y, and Z
        X = vol.shape[2]
        Y = vol.shape[1]
        Z = vol.shape[0]

        # Now set the pad width
        pad = ((0, Z1 - Z), (0, Y1 - Y), (0, X1 - X))

        # Pad the arrays
        new_vol = np.pad(vol, pad_width=pad, mode='minimum')

        # Expand dimension for channels (We are using 1 channel)
        new_vol = np.expand_dims(new_vol, axis=4)

        # Store new array
        shaped[i, :, :, :, :] = new_vol

    return shaped

def remove_empty(full_arrays, full_label_arrays, alpha=0):

    keepers = []

    t0 = time.time()
    n = full_arrays.shape[0]

    bern = np.random.binomial(1, alpha, n)

    for i in range(n):

        vol_lab = full_label_arrays[i]
        
        
        lab = np.max(vol_lab)
        if lab > 0 or bern[i] == 1:

            keepers.extend([i])

    thinned_arrays = np.take(full_arrays, indices = keepers, axis=0)
    thinned_label_arrays = np.take(full_label_arrays, indices = keepers, axis=0)

    print('Removing empty volumes took ' + str(time.time() - t0) + ' seconds to do!')
    print('Removed ' + str(full_arrays.shape[0] - thinned_arrays.shape[0]) + ' empty volumes! Of ' + str(full_arrays.shape[0]) + ' total volumes!')
    return thinned_arrays, thinned_label_arrays

def inverse_weight(label_volumes, a=10):

    label_volumes[label_volumes>0] = 1

    for j in range(label_volumes.shape[0]):

        lv = label_volumes[j]
        dt = ndimage.distance_transform_edt(lv)
        dt = a*np.exp(-dt/3)
        dt[dt==a] = 0
        dt[(dt>0) & (dt<1)] = 1

        label_volumes[j] = dt
        print('Finished', j+1, 'of', label_volumes.shape[0])
    return label_volumes

