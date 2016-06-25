from __future__ import absolute_import
from __future__ import print_function

from images import *
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping



nb_epoch = 1000
X_train = load_train_x_set()
Y_train = load_train_y_set()
X_test = load_test_x_set()
Y_test = load_test_y_set()

print (X_train.shape)
X_train = X_train.reshape(150, 90000)
X_test = X_test.reshape(30, 90000)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


model = Sequential()
model.add(Dense(128, input_shape=(90000,)))
model.add(Activation('sigmoid'))
model.add(Dense(81))
model.add(Activation('sigmoid'))

# definisanje SGD, lr je learning rate
sgd = SGD(lr=0.01)

# kompajliranje modela (Theano) - optimizacija svih matematickih izraza
model.compile(loss='mse', optimizer=sgd)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# obucavanje neuronske mreze
model.fit(X_train, Y_train, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# nakon obucavanje testiranje
score = model.evaluate(X_test, Y_test)
print('Test score:', score)
#print('Test accuracy:', score[1])

