import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
import lib.metrics 
import lib.load_data
import archs.R2UNet
import lib.evaluateAll


from lib.metrics import jaccard, dice_coef, jacard_dice, evaluateModel, testModel, create_dir
from lib.load_data import get_data
from archs.R2UNet import R2UNet
from lib.evaluateAll import foldTestModel


import cv2
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, model_from_json
from sklearn.model_selection import train_test_split

print('TensorFlow version: {version}'.format(version=tf.__version__))
print('Keras version: {version}'.format(version=tf.keras.__version__))
print('Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))


# Change to absolute datapath here
path_train = "../LeveeCrack_dataset"

img_height = 256
img_width = 256
ids = sorted(next(os.walk(path_train + "/images"))[2])

NUM_TEST_IMAGES = 0  
NUM_TRAIN_IMAGES = len(ids) - NUM_TEST_IMAGES

print(NUM_TEST_IMAGES, NUM_TRAIN_IMAGES)
train_images = sorted(glob(os.path.join(path_train, "images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(path_train, "masks/*")))[:NUM_TRAIN_IMAGES]

X, y = get_data(train_images, path_train, img_height, img_width, train=True)

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.15, random_state=2021)
tf.keras.backend.clear_session()

image_indices = np.arange(0,X1.shape[0] , 1)

kfold = KFold(n_splits = 10, shuffle=True, random_state=2021)
cvscores = []
fold_number = 1
for train_index, test_index in kfold.split(image_indices):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, y_train,X_test, y_test = [], [], [], []
	X_train, y_train = X1[train_index], y1[train_index]
	X_test, y_test = X1[test_index], y1[test_index]
	    
	## Create folders to save results
	experiment_name = "R2UNet"
	file_path = "../files/" + experiment_name + "_" + str(fold_number) + "/"
	csv_path = file_path + "/metrics_" + experiment_name + ".csv"

	# Model training 
	model_path = file_path 
	create_dir(model_path)
	lr = 1e-3
	batchSize = 16
	num_epochs = 200

	# Create a MirroredStrategy.
	strategy = tf.distribute.MirroredStrategy()
	print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
	with strategy.scope():
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.9)
		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		metrics =[dice_coef, jaccard, 'accuracy']
		model = R2UNet(input_filters=32, height=img_height, width=img_width, n_channels=3)
		model.compile(loss=bce, optimizer=optimizer, metrics=metrics)
		print(f'model created and compiled for fold {fold_number}')

    
	# run training
	callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coef", factor=0.25, patience=5, min_lr=15e-6, mode='max', verbose=1),
	            EarlyStopping(monitor='val_dice_coef', patience=8, mode='max'),
	            CSVLogger(csv_path),
	            ModelCheckpoint(filepath=os.path.join(model_path, 'best_model.h5'), monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True, verbose=1)]

	print('Fitting model...')
	model.fit(X_train, y_train, batch_size=batchSize, epochs=num_epochs, callbacks=callbacks, validation_data=(X_test, y_test), verbose=1)

	print('evaluate validation data')
	best_model = R2UNet(input_filters=32, height=img_height, width=img_width, n_channels=3)
	best_model.load_weights(model_path + "best_model.h5")
	exp_name = experiment_name + "_Fold" + str(fold_number)
	scores = foldTestModel(best_model, X_test, y_test, 8, exp_name, fold_number)
	print(f'Fold Scores for fold {fold_number} : {scores}')

	print('evaluate test data')
	exp_name = experiment_name + "_Test_Fold" + str(fold_number)
	scores = foldTestModel(best_model, X2, y2, 8, exp_name, fold_number)
	print(f'Fold Scores for fold {fold_number} : {scores}')
	tf.keras.backend.clear_session()

	fold_number += 1
