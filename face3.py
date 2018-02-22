import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json
import os






sns.set(style = 'white', context = 'notebook', palette = 'deep')

#DATA LOADING AND PREPARATION 
data = pd.read_csv('fer2013.csv')
training_df = data.groupby(['Usage']).get_group('Training')
test_df1 = data.groupby(['Usage']).get_group('PublicTest')
test_df2 = data.groupby(['Usage']).get_group('PrivateTest')


pixel_list_training = training_df['pixels'].tolist()
pixel_list_test1 = test_df1['pixels'].tolist()
pixel_list_test2 = test_df2['pixels'].tolist()

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

pixel_cols = DFColum()

pixel_list_training = pixelToList(pixel_list_training)
pixel_list_test1 = pixelToList(pixel_list_test1)
pixel_list_test2 = pixelToList(pixel_list_test2)

x_train_data = pd.DataFrame(pixel_list_training, columns = pixel_cols)
y_train_data = training_df['emotion']



x_test_data1 = pd.DataFrame(pixel_list_test1, columns = pixel_cols)
y_test_data1 = test_df1['emotion']

x_test_data2 = pd.DataFrame(pixel_list_test2, columns = pixel_cols)
y_test_data2 = test_df2['emotion']

x_test = pd.concat([x_test_data2, x_test_data1])
y_test = pd.concat([y_test_data2, y_test_data1])

del data
del training_df
del test_df2
del test_df1


"""
#checking for number of labels and count of each in the training data 
g = sns.countplot(y_train_data)
print(y_train_data.value_counts())
plt.show()
"""
#normalization
x_train_data = x_train_data/255.0
x_test = x_test/255.0

#reshaping to 48x48X1 matrix. 1 beacause it is greyscale
x_train_data = x_train_data.values.reshape(-1,48,48,1)
x_test = x_test.values.reshape(-1,48,48,1)

y_train_data = to_categorical(y_train_data, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)

random_seed = 2

x_train_data, x_val, y_train_data, y_val = train_test_split(x_train_data, y_train_data, test_size = 0.1, random_state = random_seed , stratify=y_train_data)

"""
g = plt.imshow(x_train_data[1][:,:,0])
plt.show()
"""

# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

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

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86


# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train_data)

history = model.fit_generator(datagen.flow(x_train_data,y_train_data, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 2, steps_per_epoch=x_train_data.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

#pickle.dumps(model)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 
plt.show()




# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = x_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((48,48)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

plt.show()

results = model.predict(x_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="emotion")

submission = pd.concat([pd.Series(range(1,x_test.shape[0]),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk...")
model.save("my_model.h5")




