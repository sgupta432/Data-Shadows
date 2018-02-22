from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2
import numpy as np 
from keras import backend as K
K.set_image_dim_ordering('tf')


def VGG_16(weights_path=None, shape=(48, 48)):
	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (48,48,1)))
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))


	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(7, activation = "softmax"))

# Define the optimizer
	optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	print ("Create model successfully")
	if weights_path:
		model.load_weights(weights_path)
	model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
	"""
	learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


	epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
	batch_size = 86
	"""
	return model

