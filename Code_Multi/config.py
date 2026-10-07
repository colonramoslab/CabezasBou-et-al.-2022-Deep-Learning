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
config['smooth'] = 1.

# Section 3 - Set these up to ensure the proper connections are set for the model to train / test.

#config['home'] = 'H:\\DeepLearning\\'   # Set this to where the master folder is
#home = os.path.join(config['home'],'AFD_Model')

#MDG 7/18/26 TEST
config['home'] = r'C:\Users\DCR lab\Documents\JB'
home = os.path.join(config['home'], 'AFD_Model')

config['inp_path'] = os.path.join(
    home,
    'Candidates_extrasets_MDG',
    'Code_Multi',
    'Model',
    'unet2D_multi_multi2-10_14_2019-fbeta_two_multi_full-10-14-13-8-0.00025-288--29.30--27.58.hdf5'
)
config['out_path'] = os.path.join(home, 'Code-Multi')

#data_path = os.path.join(home, '\\AFD_Model\\Data\\TestTeam\\')
#MDG 7/18/26 TEST
data_path = os.path.join(home, 'Candidates_extrasets_MDG')


######## HERE #########
#############################
# folders = ['test'] # folderes to process
#folders = ['AIY_Test']
#MDG 7/18/26 TEST
folders = ['2_5_20_MultiTest']

data_rfp_directories = [os.path.join(home, data_path, fold, 'Red\\') for fold in folders]
data_gcamp_directories = [os.path.join(home, data_path, fold, 'GCaMP\\') for fold in folders]

config['rfp_directories'] = data_rfp_directories
config['gcamp_directories'] = data_gcamp_directories

#data_preds = [os.path.join(home, data_path, fold, 'Predictions\\') for fold in folders]
data_preds = [os.path.join(home, data_path, fold, 'Predictions_Multi\\') for fold in folders]

config['preds'] = data_preds

# pleasants.b@gmail.com
# Cool IT guy, Brandon Pleasants
