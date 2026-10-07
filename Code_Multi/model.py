
import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Deconvolution3D, Activation, concatenate, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling3D, ZeroPadding3D, BatchNormalization, Cropping2D
from keras.optimizers import Adam
from keras.layers.merge import Add
from config import config
import glob as glob
import tifffile as tiff
import tensorflow as tf
import sys
import time
from keras import regularizers

from metrics import *
from util import *

dff = config['downsize_filter_factor']
smooth = config['smooth']
paths = config['rfp_directories']
pool_size = config['pool_size']
ilr = config['initial_learning_rate']
image_shape = config['image_shape']
im_shape = config['image_shape']
ks = config['kernel_size']
strides = config['strides']
k_init = config['init']
partition_shape = config['partition_shape']
reg_lambda = config['regularizer_lambda']


def unet2D(input_shape = image_shape, pool_size=pool_size, initial_learning_rate=ilr, ks = ks, strides = strides, reg_lambda=reg_lambda):

    inputs = Input(input_shape)
    conv11 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(inputs)
    conv12 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11)
    pool1 = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12)

    conv21 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool1)
    conv22 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv21)
    pool2 = Conv2D(int(16/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv22)

    conv31 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool2)
    conv32 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv31)
    pool3 = Conv2D(int(32/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv32)

    conv41 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool3)
    conv42 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv41)

    up5 = Conv2DTranspose(filters = int(64/dff), kernel_size=ks, strides = pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv42)
    up5 = Cropping2D(((0,1), (0,1)))(up5)

    up5 = concatenate([up5, conv32], axis=3)
    conv51 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up5)
    conv52 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv51)

    up6 = Conv2DTranspose(filters = int(32/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv52)
    up6 = Cropping2D(((0,1), (0,1)))(up6)

    up6 = concatenate([up6, conv22], axis=3)
    conv61 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up6)
    conv62 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv61)

    up7 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv62)
    up7 = Cropping2D(((0,1), (0,1)))(up7)

    up7 = concatenate([up7, conv12], axis=3)
    conv71 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up7)
    conv72 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv71)

    conv8 = Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(reg_lambda))(conv72)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    return model

def unet2D_deep(input_shape = image_shape, pool_size=pool_size, initial_learning_rate=ilr, ks = ks, strides = strides, reg_lambda=reg_lambda):

    inputs = Input(input_shape)
    conv11 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(inputs)
    conv12 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11)
    pool1 = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12)

    conv21 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool1)
    conv22 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv21)
    pool2 = Conv2D(int(16/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv22)

    conv31 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool2)
    conv32 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv31)
    pool3 = Conv2D(int(32/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv32)

    conv41 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool3)
    conv42 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv41)
    pool4 = Conv2D(int(64/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv42)

    conv51 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool4)
    conv52 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv51)

    up6 = Conv2DTranspose(filters = int(64/dff), kernel_size=ks, strides = pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv52)
    up6 = Cropping2D(((1,1), (1,1)))(up6)

    up6 = concatenate([up6, conv42], axis=3)
    conv61 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up6)
    conv62 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv61)

    up7 = Conv2DTranspose(filters = int(32/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv62)
    up7 = Cropping2D(((0,1), (0,1)))(up7)

    up7 = concatenate([up7, conv32], axis=3)
    conv71 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up7)
    conv72 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv71)

    up8 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv72)
    up8 = Cropping2D(((0,1), (0,1)))(up8)

    up8 = concatenate([up8, conv22], axis=3)
    conv81 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up8)
    conv82 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv81)

    up9 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv82)
    up9 = Cropping2D(((0,1), (0,1)))(up9)

    up9 = concatenate([up9, conv12], axis=3)
    conv91 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up9)
    conv92 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv91)

    conv10 = Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(reg_lambda))(conv92)
    act = Activation('sigmoid')(conv10)
    model = Model(inputs=inputs, outputs=act)

    return model

def unet2D_deep_multi(input_shape = image_shape, pool_size=pool_size, initial_learning_rate=ilr, ks = ks, strides = strides, reg_lambda=reg_lambda):

    inputs = Input(input_shape)
    conv11 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(inputs)
    conv12 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11)
    pool1 = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12)

    conv21 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool1)
    conv22 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv21)
    pool2 = Conv2D(int(16/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv22)

    conv31 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool2)
    conv32 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv31)
    pool3 = Conv2D(int(32/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv32)

    conv41 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool3)
    conv42 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv41)
    pool4 = Conv2D(int(64/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv42)

    conv51 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool4)
    conv52 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv51)

    up6 = Conv2DTranspose(filters = int(64/dff), kernel_size=ks, strides = pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv52)
    up6 = Cropping2D(((1,1), (1,1)))(up6)

    up6 = concatenate([up6, conv42], axis=3)
    conv61 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up6)
    conv62 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv61)

    up7 = Conv2DTranspose(filters = int(32/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv62)
    up7 = Cropping2D(((0,1), (0,1)))(up7)

    up7 = concatenate([up7, conv32], axis=3)
    conv71 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up7)
    conv72 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv71)

    up8 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv72)
    up8 = Cropping2D(((0,1), (0,1)))(up8)

    up8 = concatenate([up8, conv22], axis=3)
    conv81 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up8)
    conv82 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv81)

    up9 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv82)
    up9 = Cropping2D(((0,1), (0,1)))(up9)

    up9 = concatenate([up9, conv12], axis=3)
    conv91 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up9)
    conv92 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv91)

    conv10 = Conv2D(3, (1, 1), kernel_regularizer=regularizers.l2(reg_lambda))(conv92)
    act = Activation('softmax')(conv10)
    model = Model(inputs=inputs, outputs=act)

    return model

def unet2D_multi(input_shape = image_shape, pool_size=pool_size, initial_learning_rate=ilr, ks = ks, strides = strides, reg_lambda=reg_lambda):

    inputs = Input(input_shape)
    conv11 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(inputs)
    conv12 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11)
    pool1 = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12)

    conv21 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool1)
    conv22 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv21)
    pool2 = Conv2D(int(16/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv22)

    conv31 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool2)
    conv32 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv31)
    pool3 = Conv2D(int(32/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv32)

    conv41 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool3)
    conv42 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv41)
    pool4 = Conv2D(int(64/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv42)

    conv51 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool4)
    conv52 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv51)

    up6 = Conv2DTranspose(filters = int(64/dff), kernel_size=ks, strides = pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv52)
    up6 = Cropping2D(((1,0), (1,1)))(up6)

    up6 = concatenate([up6, conv42], axis=3)
    conv61 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up6)
    conv62 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv61)

    up7 = Conv2DTranspose(filters = int(32/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv62)
    up7 = Cropping2D(((0,1), (0,1)))(up7)

    up7 = concatenate([up7, conv32], axis=3)
    conv71 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up7)
    conv72 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv71)

    up8 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv72)
    up8 = Cropping2D(((0,1), (0,1)))(up8)

    up8 = concatenate([up8, conv22], axis=3)
    conv81 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up8)
    conv82 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv81)

    up9 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv82)
    up9 = Cropping2D(((0,1), (0,1)))(up9)

    up9 = concatenate([up9, conv12], axis=3)
    conv91 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up9)
    conv92 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv91)

    conv10 = Conv2D(3, (1, 1), kernel_regularizer=regularizers.l2(reg_lambda))(conv92)
    act = Activation('softmax')(conv10)
    model = Model(inputs=inputs, outputs=act)

    return model

def unet2D_multi_multi(input_shape = image_shape, pool_size=pool_size, initial_learning_rate=ilr, ks = ks, strides = strides, reg_lambda=reg_lambda):

    input_R = Input(input_shape)
    conv11_R = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(input_R)
    conv12_R = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11_R)
    # pool1_R = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12_R)

    input_G = Input(input_shape)
    conv11_G = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(input_G)
    conv12_G = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11_G)
    # pool1_G = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12_G)

    # merged = concatenate([pool1_R, pool1_G], axis=3)

    merged = concatenate([conv12_R, conv12_G], axis=3)
    pool1 = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(merged)

    conv21 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool1)
    conv22 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv21)
    pool2 = Conv2D(int(16/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv22)

    conv31 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool2)
    conv32 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv31)
    pool3 = Conv2D(int(32/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv32)

    conv41 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool3)
    conv42 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv41)
    pool4 = Conv2D(int(64/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv42)

    conv51 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool4)
    conv52 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv51)

    up6 = Conv2DTranspose(filters = int(64/dff), kernel_size=ks, strides = pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv52)
    up6 = Cropping2D(((1,0), (1,1)))(up6)

    up6 = concatenate([up6, conv42], axis=3)
    conv61 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up6)
    conv62 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv61)

    up7 = Conv2DTranspose(filters = int(32/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv62)
    up7 = Cropping2D(((0,1), (0,1)))(up7)

    up7 = concatenate([up7, conv32], axis=3)
    conv71 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up7)
    conv72 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv71)

    up8 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv72)
    up8 = Cropping2D(((0,1), (0,1)))(up8)

    up8 = concatenate([up8, conv22], axis=3)
    conv81 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up8)
    conv82 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv81)

    up9 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv82)
    up9 = Cropping2D(((0,1), (0,1)))(up9)

    up9 = concatenate([up9, conv12_R, conv12_G], axis=3)
    conv91 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up9)
    conv92 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv91)

    conv10 = Conv2D(3, (1, 1), kernel_regularizer=regularizers.l2(reg_lambda))(conv92)
    act = Activation('softmax')(conv10)
    model = Model(inputs=[input_R, input_G], outputs=act)

    return model

def unet2D_multi_multi2(input_shape = image_shape, pool_size=pool_size, initial_learning_rate=ilr, ks = ks, strides = strides, reg_lambda=reg_lambda):

    input_R = Input(input_shape)
    input_G = Input(input_shape)

    inp_merged = concatenate([input_R, input_G])

    conv11 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(inp_merged)
    conv12 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv11)
    pool1 = Conv2D(int(8/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv12)

    conv21 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool1)
    conv22 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv21)
    pool2 = Conv2D(int(16/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv22)

    conv31 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool2)
    conv32 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv31)
    pool3 = Conv2D(int(32/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv32)

    conv41 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool3)
    conv42 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv41)
    pool4 = Conv2D(int(64/dff), ks, strides=pool_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv42)

    conv51 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(pool4)
    conv52 = Conv2D(int(64/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv51)

    up6 = Conv2DTranspose(filters = int(64/dff), kernel_size=ks, strides = pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv52)
    up6 = Cropping2D(((1,0), (1,1)))(up6)

    up6 = concatenate([up6, conv42], axis=3)
    conv61 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up6)
    conv62 = Conv2D(int(32/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv61)

    up7 = Conv2DTranspose(filters = int(32/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv62)
    up7 = Cropping2D(((0,1), (0,1)))(up7)

    up7 = concatenate([up7, conv32], axis=3)
    conv71 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up7)
    conv72 = Conv2D(int(16/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv71)

    up8 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv72)
    up8 = Cropping2D(((0,1), (0,1)))(up8)

    up8 = concatenate([up8, conv22], axis=3)
    conv81 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up8)
    conv82 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv81)

    up9 = Conv2DTranspose(filters = int(16/dff), kernel_size=ks, strides=pool_size, kernel_regularizer=regularizers.l2(reg_lambda))(conv82)
    up9 = Cropping2D(((0,1), (0,1)))(up9)

    up9 = concatenate([up9, conv12], axis=3)
    conv91 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(up9)
    conv92 = Conv2D(int(8/dff), ks, strides=strides, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg_lambda))(conv91)

    conv10 = Conv2D(3, (1, 1), kernel_regularizer=regularizers.l2(reg_lambda))(conv92)
    act = Activation('softmax')(conv10)
    model = Model(inputs=[input_R, input_G], outputs=act)

    return model