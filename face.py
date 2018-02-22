from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import cv2
from sklearn.linear_model import Perceptron
#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
#size of images : 48 x 48
data = pd.read_csv('fer2013.csv')



'''
#converting the string greyscale values to an array.
val = data['pixels'][2].split(' ')
val = list(map(int,val))
'''




#UTILITY FUNCTIONS



#function to convert an array of greyscale pixels into an image. 
def createImage(image_array):
	im = Image.new("L", (48, 48))
	pix = im.load()
	for x in range(48):
		for y in range(48):
			tmp_val = image_array[x+y*48]
			pix[x,y] = (tmp_val % 255)
	im.save("test.png", "PNG")

#function to convert a pixel string to int valued array. 
def PixelStrToArray(row):
	im = row['pixels']
	#print(im,'\n')
	im = im.split(' ')
	
	im = list(map(int,im))
  
	return im

def chnageDataPixels(data):

	#data['Pixels'] = data.apply(PixelStrToArray, axis = 1)

	for x in range(data.shape[0]):
		col = data.loc[x,'pixels']

		col = col.split(' ')
		col = list(map(int,col))
		#data.loc[x,'pixels'] = col

		for y in range(2304):
			data.iloc[[x],data.columns.get_loc(col_name)] = col[y]


'''
def arrayToMatrix(row):
	a = np.zeros([48,48])
	#im = row['pixels']
	for x in range(48):
		for y in range(48):
			a[x,y] = row[x+ y*48]
			#np.append(a[x],row[x+y*48])
	return a.astype(int)

def listToLists(row):
	L = row['Pixels']
	for x in range(2304):
		col = 'pixel'+str(x)
		data[col] = L[x]




def pixelToSeries(data):
	for row in data:
		listToLists(row)

'''

#chnageDataPixels(data)
training_df = data.groupby(['Usage']).get_group('Training')
for y in range(2304):
			C = 'pixel'+str(y)
			training_df.loc[:,(C)] =  pd.Series(np.zeros([training_df['pixels'].shape[0]]), index = training_df.index)
			#print(col_name)
print(training_df['pixel0'])
test_df = data.groupby(['Usage']).get_group('PublicTest')
'''
test_cols = training_df.loc[0,'pixels']
print(type(test_cols))
'''

#chnageDataPixels(training_df)
#chnageDataPixels(test_df)
#pixelToSeries(training_df)
print(training_df.head())
#print(type(training_df['pixels'][0]))
"""
for x in range(training_df.shape[0]):
	if(len(training_df['pixels'][x]) != 2304):
		print(x,'\n')
"""

#print(training_df.head())
#print(test_df.head())
"""
x_train = training_df['Pixels']
y_train = training_df['emotion']

x_test = test_df['Pixels']

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
Y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
"""









