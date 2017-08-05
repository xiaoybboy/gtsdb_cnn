import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ProgbarLogger
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

CONTINUE = 0

if CONTINUE <= 0:
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu', input_shape=(40,40,3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(5,5), padding="same", activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
else:
    model = keras.models.load_model('models/gtsrb2/gtsrb2-{:d}.hdf5'.format(CONTINUE))

gen = ImageDataGenerator(rescale=1./255)
traingen = gen.flow_from_directory('../dataset/GTSRB/Train', target_size=(40,40), batch_size=100)
valgen   = gen.flow_from_directory('../dataset/GTSRB/Val'  , target_size=(40,40), batch_size=100)

model.fit_generator(
    traingen,
    initial_epoch = CONTINUE,
    steps_per_epoch = 360,
    epochs = 100,
    validation_data = valgen,
    validation_steps = 32,
    verbose = 1,
    callbacks = [
        ProgbarLogger(count_mode='steps'),
        ModelCheckpoint('models/gtsrb2/gtsrb2-{epoch}.hdf5', verbose=1, save_best_only = True),
        TensorBoard(log_dir='tblogs/gtsrb2/', write_graph=True, write_grads=True, write_images=True),
        EarlyStopping(patience=5, verbose=1),
    ],)
