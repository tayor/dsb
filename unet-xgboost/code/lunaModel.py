import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

import lunaIterator
from lunaIterator import lunaIterator

smooth = 1.
img_height = 512
img_width = 512


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1,img_height, img_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-6), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_model(csv_path, scan_path, results_path, batch_size=4, num_epoch=8, load_weights=False):

    # all nodules
    df = pd.read_csv(csv_path)
    num_ids = len(df)
    num_train = int(num_ids * 0.75)
    num_val = num_ids - num_train
    df_train = df[:num_train]
    df_val = df[num_train:]

    # only largest nodules
	# idx = df.groupby('seriesuid')['diameter_mm'].transform(max) == df['diameter_mm']
	# df_max = df[idx]
	# num_ids = len(df_max)
	# num_train = int(num_ids * 0.75)
	# num_val = num_ids - num_train
	# df_train = df_max[:num_train]
	# df_val = df_max[num_train:]

    train_gen = lunaIterator(scan_path, df_train, batch_size=batch_size, shuffle=True)
    val_gen = lunaIterator(scan_path, df_val, batch_size=batch_size, shuffle=False)

    model = get_unet()

    if load_weights:
    	model.load_weights(results_path+'weights_1e-6_01_-0.608.hdf5')

    model_checkpoint = ModelCheckpoint(results_path+'weights_1e-6_{epoch:02d}_{val_loss:.3f}.hdf5',
                                         monitor='val_loss', save_best_only=False, save_weights_only=True)

    model.fit_generator(train_gen, num_train, nb_epoch=num_epoch, validation_data=val_gen,
                    nb_val_samples=num_val, callbacks=[model_checkpoint])


if __name__ == '__main__':

	data_path = '../../../data/luna16/'
	scan_path = data_path+'scans/'
	csv_path = data_path+'csvfiles/annotations.csv'
	results_path = data_path+'results/'

	train_model(csv_path, scan_path, results_path, num_epoch=5, load_weights=False)
