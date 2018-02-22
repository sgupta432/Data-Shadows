from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import cv2
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import math
from sklearn import decomposition 
from sklearn import datasets

from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
#size of images : 48 x 48
data = pd.read_csv('fer2013.csv')
training_df = data.groupby(['Usage']).get_group('Training')
test_df = data.groupby(['Usage']).get_group('PublicTest')
pixel_list_training = training_df['pixels'].tolist()
pixel_list_test = test_df['pixels'].tolist()


def pixelToList(pixel_list):
	L = []
	for row in pixel_list:
		row = row.split(' ')
		row = list(map(int,row))
		L.append(row)
	return L

def DFColum():
	col_list = []
	for x in range(2304):
		col_list.append('pixel'+str(x))

	return col_list

pixel_cols = DFColum();

pixel_list_training = pixelToList(pixel_list_training)
pixel_list_test = pixelToList(pixel_list_test)

train_data = pd.DataFrame(pixel_list_training, columns = pixel_cols)
#pixel_data.insert(loc = 0 , column = 'emotion', value = training_df['emotion'])
#pixel_data['emotion'] = training_df['emotion']
test_data = pd.DataFrame(pixel_list_test, columns = pixel_cols)


#pixel_data_test['emotion'] = test_df['emotion']
#pixel_data_test.insert(loc = 0 , column = 'emotion', value = test_df['emotion'])
"""
plt.hist(pixel_data['emotion'])
plt.title("frequency histohram of emotions in training data")
plt.xlabel("Emotion Value")
plt.ylabel("Frequency")
plt.show()
#Note to self: From this plot it appears that very few disgust images are available
"""

#print(pixel_data.shape) #size of training data is 28709 x 2305 - 2304 pixels for the image and 1 for the emotion label
#print(pixel_data.head())
#print(pixel_data_test.head())

def plotTrainingImages(train_data):
	f,ax = plt.subplots(5,5)
	for i in range(0,25):
		data = train_data.iloc[i, 0:2304].values
		nrows, ncols = 48, 48
		grid = data.reshape((nrows,ncols))
		n = math.ceil(i/5)-1
		m=[0,1,2,3,4]*5
		ax[m[i-1],n].imshow(grid)
	plt.show()

##Normalizing the training data set ##

y_train = training_df['emotion']

train_data = train_data/255
test_data = test_data/255

#train_data['emotion'] = y_train

##PCA decomposition
pca = decomposition.PCA(n_components = 200) #find the first 200 PCs
pca.fit(train_data)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel(' % of variance explained')
#plt.show()

pca = decomposition.PCA(n_components = 50)
pca.fit(train_data)

PCtrain = pd.DataFrame(pca.transform(train_data))
PCtrain['emotion'] = y_train

PCtest = pd.DataFrame(pca.transform(test_data))


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
x = PCtrain[0]
y = PCtrain[1]
z = PCtrain[2]

colors = [int(i%6) for i in PCtrain['emotion']]
ax.scatter(x, y, z, c = colors, marker = 'o', label = colors)

ax.set_xlabel('PC1')
ax.set_label('PC2')
ax.set_label('PC3')

#plt.show()

"""
Y = PCtrain['emotion'][0:22000]
X = PCtrain.drop('emotion', axis = 1)[0:22000]

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (3500,), random_state = 1)
clf.fit(X,Y)

print(clf)

predicted = clf.predict(PCtrain.drop('emotion', axis = 1)[22000:])
expected = PCtrain['emotion'][22000:]

print("Classification report fpr classifier %s: \n %s \n" % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix: \n%s" % metrics.confusion_matrix(expected, predicted))

output = pd.DataFrame(clf.predict(PCtest), columns =['emotion'])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)
output['ImageId']=output['ImageId']+1
output.to_csv('output.csv', index=False)
"""

print(test_df['emotion'])




