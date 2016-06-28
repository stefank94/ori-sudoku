# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:42:40 2016

@author: Sebastijan
"""
from images import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

model = Sequential()

def learn():

    nb_classes = 10
    
    save_weights = False #da li da sacuvamo tezine
    test_only = True  # da li samo vec obucili mrezu
    
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)
    #stampa primere slika
    '''for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i+19], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[i]))'''
    
    '''Our neural-network is going to take a single vector for each training example, 
    so we need to reshape the input so that each 28x28 image becomes a single 784 dimensional vector.
     We'll also scale the inputs to be in the range [0-1] rather than [0-255]'''
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
        
    #ucitavanje nasih slika    
    if test_only==False:
        tables = get_images_for_learning()
        for j in xrange(len(tables)):
            x = tables[j]
            print(j)
            for i in xrange(len(x)):
                X_train = np.vstack([X_train, x[i][0]])
        
    
    #Modify the target matrices to be in the one-hot format
    ''' 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
        1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0]
        2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0]
        etc.'''
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    #ucitavanje tacnih klasa nasih slika    
    if test_only==False:
        tables = get_classes()
        for i in xrange(len(tables)):
            y = tables[i]
            y = np_utils.to_categorical(y, nb_classes)
            Y_train = np.vstack([Y_train, y])
    
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)
    
    #3 layer fully connected network
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                                  # of the layer above. Here, with a "rectified linear unit",
                                  # we clamp all values below 0 to 0.
    
    model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                     # ensures the output is a valid probaility distribution, that is
                                     # that its values are all non-negative and sum to 1.
    
    
    
    # ucitavanje snimljenih tezina iz prethodnog obucavanja
    if test_only:
        model.load_weights('weights/tezine.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=["accuracy"])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=["accuracy"])
        model.fit(X_train, Y_train,
              batch_size=128, nb_epoch=20, verbose=0,
              validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    if save_weights:
        model.save_weights('weights/tezine.hdf5', overwrite=True)
        
def predict(path):

    matrix = np.zeros(81,dtype=np.uint8)
    numbers = get_one_image(path)
    
    for i in xrange(0,81):
        if sum(numbers[i][0]) < 5:
            matrix[i] = 0
        else:
            predicted = model.predict_classes(numbers[i], verbose = 0)
            matrix[i] = predicted
        #predict = model.predict_proba(get_one_image())
        #print(predict)
        #print(max(predict[0]))
    matrix = matrix.reshape((9, 9))
    return matrix

'''
#provera sta je dobro predvideo a sta nije(stampa primere)
# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
'''

