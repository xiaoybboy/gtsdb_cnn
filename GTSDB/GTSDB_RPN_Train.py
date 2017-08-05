import keras
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.callbacks import ProgbarLogger, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

import numpy as np
from skimage.io import imread

################################# BASE LAYERS #################################

def vgg(input_layer, trainable=False):
    vgg19 = keras.applications.vgg19.VGG19(include_top=False)
    vgg19.trainable = trainable
    return vgg19(input_layer)

def base(input_layer):
    return Sequential([
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu', input_shape=(800,1360,3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(5,5), padding="same", activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
    ])(input_layer)


################################# RPN LAYERS #################################

def rpn(base, k):
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_initializer=keras.initializers.RandomNormal(0.0, 0.01), name='rpn_conv_1')(base)
    return(
        Conv2D(k,   (1, 1), activation='sigmoid',
               kernel_initializer=keras.initializers.RandomNormal(0.0, 0.01), name='rpn_cls')(x),
        Conv2D(k*4, (1, 1), activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(0.0, 0.01), name='rpn_regr')(x))


################################# BOXES #################################


def get_anchors(rows, cols, sizes, stride):
    # Generate 1:1 anchor boxes
    # anc_num = len(sizes) [= k in the frcnn paper]
    # shape = (rows, cols, anc_num, 4)
    return np.expand_dims(np.tile(np.indices((rows,cols)).transpose((1,2,0)) * stride + .5, 2), axis=2).repeat(len(sizes), axis=2) \
            + np.repeat(np.expand_dims(np.array(sizes), axis=1), 4, axis=1) * [-.5, -.5, .5, .5]


def intersect(b1, b2):
    # b1.shape = (rows, cols, anc_num, 4)
    # b2.shape = (4,)
    m = np.minimum(b1,b2)
    M = np.maximum(b1,b2)
    h = np.maximum(m[...,2] - M[...,0], 0)
    w = np.maximum(m[...,3] - M[...,1], 0)
    return w*h

def union(b1, b2, iarea):
    # b1.shape = (rows, cols, anc_num, 4)
    # b2.shape = (4,)
    a1 = (b1[...,2]-b1[...,0]) * (b1[...,3]-b1[...,1])
    a2 = (b2[...,2]-b2[...,0]) * (b2[...,3]-b2[...,1])
    return a1 + a2 - iarea

def iou(b1, b2):
    # b1.shape = (rows, cols, anc_num, 4)
    # b2.shape = (4,)
    iarea = intersect(b1, b2)
    uarea = union(b1, b2, iarea)
    return iarea/uarea

def coords2param(ancs, gtbs):
    # Convert absolute coords to parametrized params (see frcnn paper)
    # ancs.shape = gtbs.shape = (rows, cols, anc_num, 4)
    # box = [y1,x1,y2,x2]
    
    wa = ancs[...,3] - ancs[...,1]
    ha = ancs[...,2] - ancs[...,0]
    tx = (gtbs[...,1] - ancs[...,1]) / wa
    ty = (gtbs[...,1] - ancs[...,1]) / ha
    tw = np.log((gtbs[...,3] - gtbs[...,1]) / wa)
    th = np.log((gtbs[...,2] - gtbs[...,0]) / ha)
    
    # shape = (row, cols, anc_num, 4)
    return np.stack([tx,ty,tw,th], axis=-1)

def anchors_vs_gt(ancs, gtbs, lo, hi):
    # ancs.shape = (rows, cols, anc_num, 4)
    # gtbs.shape = (gtb_num, 4)
    
    # ious.shape = (rows, cols, anc_num, gtb_num)
    ious = np.stack([iou(ancs, gtb) for gtb in gtbs], axis=-1)
    # best.shape = (gtb_num,)
    best = ious.reshape((-1, gtbs.shape[0])).max(axis=0)
    
    # box_pos.shape = box_neg.shape = (rows, cols, anc_num)
    box_pos = np.logical_or(ious.max(axis=-1) >= hi, np.logical_and(ious == best, best > 0).any(axis=-1))
    box_neg = ious.max(axis=-1) <= lo
    
    # hard_pos = anchor boxes with iou >= 0.7 with any gt box
    # soft_pos = anchor boxes with highest iou with a gt box
    # hard_neg = anchor boxes with iou <= 0.3 with all gt boxes
    ## print("\thard_pos = {:d}".format(np.sum(ious.max(axis=-1) >= hi)))
    ## print("\tsoft_pos = {:d}".format(np.sum(np.logical_and(ious == best, best > 0).any(axis=-1))))
    ## print("\thard_neg = {:d}".format(np.sum(ious.max(axis=-1) <= lo)))
    
    # best_gt.shape = (rows, cols, anc_num, 4)
    best_gt = np.take(gtbs, ious.argmax(axis=-1), axis=0)
    
    return box_pos, box_neg, coords2param(ancs, best_gt)

def filter_boxes(pos, neg, num):
    # Only use num boxes, with pos:neg at most 1:1 unless pos < num/2
    p_num = pos[pos].shape[0]
    n_num = neg[neg].shape[0]
    if p_num > num/2:
        pos[np.vsplit(np.vstack(np.where(pos))[:,np.random.choice(p_num, p_num-num//2, replace=False)], pos.ndim)] = False
        p_num = num//2
    if n_num + p_num > num:
        neg[np.vsplit(np.vstack(np.where(neg))[:,np.random.choice(n_num, n_num-num+p_num, replace=False)], neg.ndim)] = False
    return pos, neg

################################# LOSSES #################################

def rpn_regr_loss(num_ancs):
    def loss(ytrue, ypred, ptrue):
        # ytrue.shape = (rows, cols, num_ancs * 4)
        # ypred.shape = (rows, cols, num_ancs * 4)
        # ptrue.shape = (rows, cols, num_ancs * 4)
        # ancs.shape  = (rows, cols, num_ancs * 4)
        
        dy = ytrue - ypred
        sw = K.cast(K.less(K.abs(dy), 1), dtype=K.floatx())
        r1 = sw*dy*dy*.5
        r2 = (1-sw)*(K.abs(dy)-.5)
        return K.sum((r1+r2) * ptrue)
    return lambda ytrue, ypred: \
        loss(ytrue[...,4*num_ancs:],
             ypred,
             ytrue[...,:4*num_ancs])

def rpn_cls_loss(num_ancs):
    def loss(postrue, negtrue, ppred):
        # Add epsilon = 1e-4 to prevent log(0)
        return K.sum(- postrue * K.log(1e-4 + ppred)
                     - negtrue * K.log(1e-4 + 1-ppred))
    return lambda ptrue, ppred: loss(ptrue[...,:num_ancs], ptrue[...,num_ancs:], ppred)

################################# DATA #################################

def datagen(start, stop, ancs=None, shuffle=True):
    """Generator for GTSDB dataset
        Args:
            start, stop = range of images to use
            ancs = anchor boxes to use. Use get_anchors() to generate these.
                Dimensions should be (imageheight/basenet_stride, imagewidht/basenet_stride, k, basenet_stride)
                Defaults to be get_anchors(200, 340, [16,24,32], 4)
            shuffle = whether to shuffle the data
    """
    if ancs is None:
        ancs = get_anchors(200, 340, [16,24,32], 4)
    csv = np.loadtxt('../dataset/PNG_train/gt.txt', delimiter=',', converters = {0: lambda x:x[:-4]}, dtype=np.int32)
    
    idx = np.arange(start, stop)
    
    for i in idx:
        temp = csv[csv[:,0] == i]
        temp = temp[:,[2,1,4,3]]
        # temp.shape = (gtb_num, 4)
        
        ## print("fname = ../dataset/PNG_train/{:05d}.png".format(i))
        pos, neg, gtbs = anchors_vs_gt(ancs, temp, .3, .7)
        gtbs = gtbs.reshape((gtbs.shape[0], gtbs.shape[1], -1))
        # pos.shape = neg.shape = (rows, cols, anc_num)
        # gtbs.shape = (row, cols, anc_num * 4)
        pos, neg = filter_boxes(pos, neg, 256)
        
        # x_img.shape  = (1, imgh, imgw, imgchannels)
        # y_cls.shape  = (row, cols, anc_num * 2)
        # y_regr.shape = (row, cols, anc_num * 8)
        x_img  = np.expand_dims(imread('../dataset/PNG_train/{:05d}.png'.format(i)), 0)
        y_cls  = np.expand_dims(np.concatenate((pos, neg), axis=-1).astype(np.int32), 0)
        y_regr = np.expand_dims(np.concatenate((pos.repeat(4, axis=-1), gtbs), axis=-1), 0)
        
        yield (x_img, [y_cls, y_regr])

################################# MODEL #################################

# Find existing model
import os, re
temp = [re.compile('gtsdb_rpn-(\d+)\.hdf5').match(fn) for fn in os.listdir('models/gtsdb_rpn/')]
temp = [int(m.group(1)) for m in temp if m is not None]

# Load model if it exsits. Else build new model
if len(temp) == 0:
    inp = Input(shape=(800,1360,3))
    model = Model(inputs=inp, outputs=rpn(base(inp), 3))
    model.compile(optimizer='sgd', loss={'rpn_cls': rpn_cls_loss(3), 'rpn_regr': rpn_regr_loss(3)})
else:
    model = keras.models.load_model('models/gtsdb_rpn/gtsdb_rpn-{:d}.hdf5'.format(max(temp)))

################################# TRAIN #################################

model.fit_generator(
    datagen(0, 600, shuffle=True),
    steps_per_epoch = 600,
    epochs = 100,
    validation_data = datagen(600, 900),
    validation_steps = 300,
    verbose = 1,
    callbacks = [
        ProgbarLogger(count_mode='steps'),
        ModelCheckpoint('models/gtsdb_rpn/gtsdb_rpn-{epoch}.hdf5', verbose=1, save_best_only = True),
        TensorBoard(log_dir='tblogs/gtsdb_rpn/', write_graph=True, write_grads=True, write_images=True),
        EarlyStopping(patience=5, verbose=1),
    ],)

