import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ProgbarLogger
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load trained model and weights at the last iteration of training
model = keras.models.load_model('models/gtsrb1/gtsrb1-last.hdf5')

gen = ImageDataGenerator(rescale=1./255)
testgen = gen.flow_from_directory('../dataset/GTSRB/Test', classes=None, target_size=(40,40), batch_size=1000, shuffle=True)

# Use only first 1000 images for this
# Number of images adjustable on line above
batch = testgen.next()[0]
preds = model.predict_on_batch(batch)
preds = np.argmax(preds, axis=-1)

cols = 4
for cls in range(43):
    imgs = batch[preds==cls]
    num = imgs.shape[0]
    rows = num//cols + 1
    fig = plt.figure()
    fig.suptitle('Class {:d}'.format(cls), fontsize=36)
    fig.set_size_inches((16, 16 / cols * rows))
    for j in range(num):
        plt.subplot(rows, cols, j+1)
        plt.imshow(imgs[j])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

