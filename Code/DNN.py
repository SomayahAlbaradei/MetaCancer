#!/usr/bin/env python
# coding: utf-8

# after completing the CVAE training process, 
# the encoder part is fed to a DNN classifier.
# We used the latent vector as input to train the classifier to make prediction.


from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=3, monitor='val_loss')

# create model
# Building the DNN requires configuring the layers of the model, then compiling the model. 
# First we stack a few layers together using. 
# Next we configure the loss function, optimizer, and metrics to monitor.

model = Sequential()
model.add(Dense(60, input_dim=100, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='tanh', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

#Training the DNN model 

#Feed the training data to the model, the model learns to associate features and labels.

history =model.fit(x_train, y_train,
        epochs=100,
        batch_size=8,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping])

