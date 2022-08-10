import os
import sys

config = dict()


config['padding'] = 0
config['partition_shape'] = 0
config['nvcs'] = 0 


# Sections 1 - Model Architecture:

config['pool_size'] = (2, 2)
config['kernel_size'] = (3, 3)
# config['image_shape'] = (200, 200, 1)
config['image_shape'] = (2000, 1000, 1)
config['n_labels'] = 1
config['strides'] = 1
config['downsize_filter_factor'] = 1
config['regularizer_lambda'] = 0.0005
config['init'] = 'he_normal'

config['initial_learning_rate'] = 1.

# Section 3 - Set these up to ensure the proper connections are set for the model to train / test.

config['home'] = 'E:\\FreelyMoving\\DeepLearning\\'   # Set this to where the master folder is
home = os.path.join(config['home'],'AFD_Model\\')


config['inp_path'] = os.path.join(home, 'Code', 'Model', 'unet2D_multi-7_12_2019-fbeta_two_multi_full-8-7-11-19-0.0005-194--15.76--14.68.hdf5')
config['out_path'] = os.path.join(home, 'Code')

data_path = os.path.join(home, 'Data\\9_19_19\\')

######## HERE #########
#############################
folders = ['GCY8Mix_Tc25_1__3'] # folderes to process

data_raw_directories = [os.path.join(home, data_path, fold, 'Red\\') for fold in folders]

config['raw_directories'] = data_raw_directories

data_preds = [os.path.join(home, data_path, fold, 'Predictions\\') for fold in folders]

config['preds'] = data_preds

# pleasants.b@gmail.com
# Cool IT guy, Brandon Pleasants