
import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Deconvolution3D, Activation, concatenate, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling3D
from keras.optimizers import Adam
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import Add
from keras.layers.advanced_activations import PReLU
from keras.losses import binary_crossentropy
from config import config
import glob as glob
import tifffile as tiff
import tensorflow as tf
import sys
import time

########################

# Metrics - Binary

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def precision(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    true_positives = K.sum(y_true_f*y_pred_f)
    predicted_positives = K.sum(y_pred_f)

    prec = true_positives / (predicted_positives + K.epsilon())

    return(prec)

def recall(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(y_true_f*y_pred_f)
    actual_positives = K.sum(y_true_f) 

    rec = true_positives / (actual_positives + K.epsilon())

    return rec

def fbeta_half(y_true, y_pred, beta=0.5):

    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)

    coef = 1 + beta**2
    num = prec*rec
    dem = (beta**2 * prec) + rec

    return coef*num / (dem + K.epsilon())

def fbeta_quarter(y_true, y_pred, beta=0.25):

    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)

    coef = 1 + beta**2
    num = prec*rec
    dem = (beta**2 * prec) + rec

    return coef*num / (dem + K.epsilon())

def fbeta_two(y_true, y_pred, beta=2.0):

    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)

    coef = 1 + beta**2
    num = prec*rec
    dem = (beta**2 * prec) + rec

    return coef*num / (dem + K.epsilon())

def hard_dice(y_true, y_pred, thresh=.9999):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.cast(K.greater(y_true, 0), K.floatx())
    y_pred_f = K.cast(K.greater(K.clip(y_pred, 0, 1), thresh), K.floatx())
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def hard_precision(thresh=0.998):
    def precision1(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = thresh
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.

        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)

        y_true = K.cast(K.greater(y_true, 0), K.floatx())
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())

        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision1

def hard_recall(thresh=0.998):
    def recall1(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = thresh
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.

        y_true = K.cast(K.greater(y_true, 0), K.floatx())
        y_pred = K.cast(K.greater(y_pred, threshold_value), K.floatx())

        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall1

# Metrics - Multi Class

def dice_coef_multilabel_full(y_true, y_pred):

    dice = dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0]) + dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1]) + dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])
    return dice

def dice_coef_multilabel(y_true, y_pred):

    dice = dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1]) + dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])
    return dice

def dice_coef0(y_true, y_pred):
    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef1(y_true, y_pred):
    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef2(y_true, y_pred):
    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.flatten(y_pred[:,:,:,2])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision0(y_true, y_pred):

    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    
    true_positives = K.sum(y_true_f*y_pred_f)
    predicted_positives = K.sum(y_pred_f)

    prec = true_positives / (predicted_positives + K.epsilon())

    return(prec)

def recall0(y_true, y_pred):
    
    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])

    true_positives = K.sum(y_true_f*y_pred_f)
    actual_positives = K.sum(y_true_f) 

    rec = true_positives / (actual_positives + K.epsilon())

    return rec

def precision1(y_true, y_pred):

    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,1])
    
    true_positives = K.sum(y_true_f*y_pred_f)
    predicted_positives = K.sum(y_pred_f)

    prec = true_positives / (predicted_positives + K.epsilon())

    return(prec)

def recall1(y_true, y_pred):
    
    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,1])

    true_positives = K.sum(y_true_f*y_pred_f)
    actual_positives = K.sum(y_true_f) 

    rec = true_positives / (actual_positives + K.epsilon())

    return rec

def precision2(y_true, y_pred):

    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.flatten(y_pred[:,:,:,2])
    
    true_positives = K.sum(y_true_f*y_pred_f)
    predicted_positives = K.sum(y_pred_f)

    prec = true_positives / (predicted_positives + K.epsilon())

    return(prec)

def recall2(y_true, y_pred):
    
    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.flatten(y_pred[:,:,:,2])

    true_positives = K.sum(y_true_f*y_pred_f)
    actual_positives = K.sum(y_true_f) 

    rec = true_positives / (actual_positives + K.epsilon())

    return rec


def fbeta_two_multilabel_full(y_true, y_pred):

    y_true_0 = y_true[:,:,:,0]
    y_true_1 = y_true[:,:,:,1]
    y_true_2 = y_true[:,:,:,2]

    y_pred_0 = y_pred[:,:,:,0]
    y_pred_1 = y_pred[:,:,:,1]
    y_pred_2 = y_pred[:,:,:,2]

    fb_2_0 = fbeta_two(y_true_0, y_pred_0)
    fb_2_1 = fbeta_two(y_true_1, y_pred_1)
    fb_2_2 = fbeta_two(y_true_2, y_pred_2)
    
    fb = fb_2_0 + fb_2_1 + fb_2_2

    return fb

# Losses - Binary

def prec_loss(y_true, y_pred):

    return -precision(y_true, y_pred)

def prec_rec_loss(y_true, y_pred):

    return -(precision(y_true, y_pred) + recall(y_true,y_pred))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def fbeta_half_loss(y_true, y_pred):

    return -fbeta_half(y_true, y_pred, beta=0.5)

def fbeta_quarter_loss(y_true, y_pred):

    return -fbeta_quarter(y_true, y_pred, beta=0.25)

def fbeta_two_loss(y_true, y_pred):

    return - fbeta_two(y_true, y_pred, beta=2.0)

# Losses - Multi Class

def dice_coef_multilabel_full_loss(y_true, y_pred):

    w0 = 1
    w1 = 2
    w2 = 10

    dice = 0
    dice += - (w0*dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0]) + w1*dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1]) + w2*dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2]))
   
    return dice

def dice_coef_multilabel_loss(y_true, y_pred, numLabels=3):

    dice = 0
    dice += - (dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1]) + dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2]))
   
    return dice

def wce_loss(y_true,y_pred):

    eps = K.epsilon()

    # y_pred: [B, Z, Y, X, 3] {0,1}
    # y_true: [B, Z, Y, X, 3] [0,1]

    y_pred0 = K.clip(K.flatten(y_pred[:,:,:,:,0]), eps, 1 - eps)
    y_true0 = K.flatten(y_true[:,:,:,:,0])

    y_pred1 = K.clip(K.flatten(y_pred[:,:,:,:,1]), eps, 1 - eps)
    y_true1 = K.flatten(y_true[:,:,:,:,1])

    y_pred2 = K.clip(K.flatten(y_pred[:,:,:,:,2]), eps, 1 - eps)
    y_true2 = K.flatten(y_true[:,:,:,:,2])

    #dims = K.int_shape(y_pred)
    #total = tf.size(y_pred)

    #w0 = total / K.sum(y_true0)
    #w1 = total / K.sum(y_true1)
    #w2 = total / K.sum(y_true2)
    
    w0 = 1.
    w1 = 10.
    w2 = 1.

    ce0 = -w0*K.mean( y_true0*K.log(y_pred0) + (1-y_true0)*K.log(1-y_pred0))
    ce1 = -w1*K.mean( y_true1*K.log(y_pred1) + (1-y_true1)*K.log(1-y_pred1))
    ce2 = -w2*K.mean( y_true2*K.log(y_pred2) + (1-y_true2)*K.log(1-y_pred2))

    return ce0 + ce1 + ce2

def bce_and_dice_loss(y_true, y_pred):

    y_true_f = K.clip(K.flatten(y_true), K.epsilon(), 1-K.epsilon())
    y_pred_f = K.clip(K.flatten(y_pred), K.epsilon(), 1-K.epsilon())
    ce = K.mean(y_true_f*K.log(y_pred_f) + (1-y_true_f)*K.log(1-y_pred_f))
    intersection = K.sum(y_true*y_pred)
    dc = (2*intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    
    return -ce - dc

def fbeta_two_multilabel_loss(y_true, y_pred):

    y_true_0 = y_true[:,:,:,0]
    y_true_1 = y_true[:,:,:,1]
    y_true_2 = y_true[:,:,:,2]

    y_pred_0 = y_pred[:,:,:,0]
    y_pred_1 = y_pred[:,:,:,1]
    y_pred_2 = y_pred[:,:,:,2]

    fb_2_0 = fbeta_two(y_true_0, y_pred_0)
    fb_2_1 = fbeta_two(y_true_1, y_pred_1)
    fb_2_2 = fbeta_two(y_true_2, y_pred_2)

    w0 = 1
    w1 = 2
    w2 = 15
    
    fb = -(w0*fb_2_0 + w1*fb_2_1 + w2*fb_2_2)

    return fb


