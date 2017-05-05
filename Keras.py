##NN, Keras
#TensorFlow, Theano combine
#pip install keras

##3 backend; kernel from either tensorflow or theano
#theano or tensorflow basis NN

#m1 find the folder 
#~/.keras/keras.json
'''
{...
	"backend":"tensorflow" <<change to "theano"
}
'''

#m2 temp change
# import os
# os.environ['KERAS_BACKEND']='theano'
# import keras



##4 regression
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential #sequ layer
from keras.layers import Dense #all connect layer
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
#if second layer add could omit input_dim as 1
#model.add(Dense(output_dim=1, )) #will enough

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')
#MSE, stocha gd

# training
print('Training -----------')
for step in range(301):
 cost = model.train_on_batch(X_train, Y_train) #base on batch, equal batch
 if step % 100 == 0:
     print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40) #last 40 examples
print('test cost:', cost)
W, b = model.layers[0].get_weights() #w=0.5, b=2 above
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()



##5 classification
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10) #numpy script into (1000000000) dummy
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your neural net
model = Sequential([
 Dense(32, input_dim=784), #hidden 32 layers
 Activation('relu'),
 Dense(10),                #only need 10 output
 Activation('softmax'),    
])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,    #or 'rmsprop' default
           loss='categorical_crossentropy',
           metrics=['accuracy']) #or 'cost'

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)



##6 CNN(deep learning)
#not only pixel, but patch
#convolution, patch more depth
#pooling, compress 



##7 CNN
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    nb_filter=32,           #32filter in one figure, 32 features(layers)
    #kernel_size=(5,5),               #width
    nb_row=5, nb_col=5,               #height
    padding='same',     # Padding method, 
    dim_ordering='th',      # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
    input_shape=(1,         # channels
                 28, 28,)   # height & width
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D( #AvgPooling2D
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',    # Padding method
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, 5, padding='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())    #above 3d into 1d
model.add(Dense(1024))  #from 3136 into 1024 knots
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))    #from 1024 into 10 outputs
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch=1, batch_size=32,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)




