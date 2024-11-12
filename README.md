# EX-04 : Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
![image](https://github.com/user-attachments/assets/016402a5-6816-45a0-8ed0-000bf929b5be)

## Neural Network Model

![image](https://github.com/user-attachments/assets/8be0a18a-d61f-43f5-b853-640ffc477c06)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot
### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input

## PROGRAM

### Name: ARUN KUMAR SUKDEV CHAVAN
### Register Number: 212222230013

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[5]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[600]
plt.imshow(single_image,cmap='gray')

y_train_onehot[600]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')
print(metrics.head())

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')
metrics[['accuracy','val_accuracy']].plot()

print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')
metrics[['loss','val_loss']].plot()

print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')
print(confusion_matrix(y_test,x_test_predictions))

print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')
print(classification_report(y_test,x_test_predictions))

img = X_test_scaled[5]
img.shape
plt.imshow(img,cmap='gray')

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_tensor.shape
img_28 = tf.image.resize(img_tensor,(28,28))
#img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
print('''ARUN KUMAR SUKDEV CHAVAN
212222230013''')

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/3b0c143e-e1c4-4f01-a30b-7e3622d87b06)



![image](https://github.com/user-attachments/assets/c1ddba73-7471-4bd1-a006-cba05e52536f)



![image](https://github.com/user-attachments/assets/4d5e8c24-ed6e-422d-b33f-5109a9a9b358)



### Classification Report

![image](https://github.com/user-attachments/assets/92fa39d5-3094-406c-874a-3f89460209e6)



### Confusion Matrix

![image](https://github.com/user-attachments/assets/585eff0f-e801-4a21-ab98-19e665c29dc1)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/8ad9a18a-9cd9-4fef-a44b-bc8dd774b2b5)



## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
