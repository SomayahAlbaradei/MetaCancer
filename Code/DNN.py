#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=3, monitor='val_loss')

# create model
model = Sequential()
model.add(Dense(60, input_dim=100, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='tanh', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

history =model.fit(x_train, y_train,
        epochs=100,
        batch_size=8,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping])

