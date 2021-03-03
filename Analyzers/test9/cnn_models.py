import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model, clone_model, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, Concatenate, Lambda, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from keras.regularizers import Regularizer
from keras.constraints import Constraint
from keras import initializers, regularizers, optimizers, losses, metrics

from keras.applications.mobilenet import relu6, DepthwiseConv2D

from cnn_globals import (superstrip_size, n_zones, rows_per_zone, n_rows, n_columns, n_channels, n_classes)


def create_model(nvariables=None, lr=0.001, clipnorm=10., dropout=0.2, use_bn=True, use_dropout=False):
  input_shape = (n_rows, n_columns, n_channels)
  inputs = Input(shape=input_shape, dtype='float32')

  # Split by zones
  x = inputs
  x_zone0 = Lambda(lambda x: x[:, 0*rows_per_zone:(0+1)*rows_per_zone])(x)
  x_zone1 = Lambda(lambda x: x[:, 1*rows_per_zone:(1+1)*rows_per_zone])(x)
  x_zone2 = Lambda(lambda x: x[:, 2*rows_per_zone:(2+1)*rows_per_zone])(x)
  x_zone3 = Lambda(lambda x: x[:, 3*rows_per_zone:(3+1)*rows_per_zone])(x)
  x_zone4 = Lambda(lambda x: x[:, 4*rows_per_zone:(4+1)*rows_per_zone])(x)
  x_zone5 = Lambda(lambda x: x[:, 5*rows_per_zone:(5+1)*rows_per_zone])(x)
  x_zone6 = Lambda(lambda x: x[:, 6*rows_per_zone:(6+1)*rows_per_zone])(x)

  def _conv_block_one(x):
    x = Conv2D(
            8, (rows_per_zone,63),
            strides=(rows_per_zone,7),
            padding='valid',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    return x

  x_zone0 = _conv_block_one(x_zone0)
  x_zone1 = _conv_block_one(x_zone1)
  x_zone2 = _conv_block_one(x_zone2)
  x_zone3 = _conv_block_one(x_zone3)
  x_zone4 = _conv_block_one(x_zone4)
  x_zone5 = _conv_block_one(x_zone5)
  x_zone6 = _conv_block_one(x_zone6)

  def _conv_block_two(x):
    filters = 16
    x = DepthwiseConv2D(
            (3,3),
            strides=(1,1),
            depth_multiplier=1,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    x = Conv2D(
            filters, (1,1),
            strides=(1,1),
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    x = MaxPooling2D(
            (1,3),
            strides=(1,2),
            padding='valid')(x)
    #
    filters = 16
    x = DepthwiseConv2D(
            (3,3),
            strides=(1,1),
            depth_multiplier=1,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    x = Conv2D(
            filters, (1,1),
            strides=(1,1),
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    x = MaxPooling2D(
            (1,3),
            strides=(1,2),
            padding='valid')(x)
    #
    #filters = 16
    #x = DepthwiseConv2D(
    #        (3,3),
    #        strides=(1,1),
    #        depth_multiplier=1,
    #        padding='same',
    #        kernel_initializer='he_normal',
    #        use_bias=False,
    #        activation=None)(x)
    #x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    #x = Activation(relu6)(x)
    #x = Conv2D(
    #        filters, (1,1),
    #        strides=(1,1),
    #        padding='same',
    #        kernel_initializer='he_normal',
    #        use_bias=False,
    #        activation=None)(x)
    #x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    #x = Activation(relu6)(x)
    #x = MaxPooling2D(
    #        (1,3),
    #        strides=(1,2),
    #        padding='valid')(x)
    return x

  x_zone0 = _conv_block_two(x_zone0)
  x_zone1 = _conv_block_two(x_zone1)
  x_zone2 = _conv_block_two(x_zone2)
  x_zone3 = _conv_block_two(x_zone3)
  x_zone4 = _conv_block_two(x_zone4)
  x_zone5 = _conv_block_two(x_zone5)
  x_zone6 = _conv_block_two(x_zone6)

  # Merge zones
  x_zone01 = Concatenate()([x_zone0, x_zone1])
  x_zone12 = Concatenate()([x_zone1, x_zone2])
  x_zone23 = Concatenate()([x_zone2, x_zone3])
  x_zone34 = Concatenate()([x_zone3, x_zone4])
  x_zone45 = Concatenate()([x_zone4, x_zone5])
  x_zone56 = Concatenate()([x_zone5, x_zone6])

  def _conv_block_three(x):
    x = Conv2D(
            16, (1,1),
            strides=(1,1),
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    return x

  x_zone01 = _conv_block_three(x_zone01)
  x_zone12 = _conv_block_three(x_zone12)
  x_zone23 = _conv_block_three(x_zone23)
  x_zone34 = _conv_block_three(x_zone34)
  x_zone45 = _conv_block_three(x_zone45)
  x_zone56 = _conv_block_three(x_zone56)

  # Merge zones
  x_zone0123 = Concatenate()([x_zone01, x_zone12, x_zone23])
  x_zone1234 = Concatenate()([x_zone12, x_zone23, x_zone34])
  x_zone2345 = Concatenate()([x_zone23, x_zone34, x_zone45])
  x_zone3456 = Concatenate()([x_zone34, x_zone45, x_zone56])

  def _conv_block_four(x):
    x = Conv2D(
            16, (1,1),
            strides=(1,1),
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    return x

  x_zone0123 = _conv_block_four(x_zone0123)
  x_zone1124 = _conv_block_four(x_zone1234)
  x_zone2345 = _conv_block_four(x_zone2345)
  x_zone3456 = _conv_block_four(x_zone3456)

  # Merge zones
  x = Concatenate()([x_zone0123, x_zone1234, x_zone2345, x_zone3456])

  def _conv_block_five(x):
    x = Conv2D(
            16, (1,1),
            strides=(1,1),
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    return x

  x = _conv_block_five(x)

  #x = Flatten()(x)
  #x = Dense(32, activation='relu')(x)
  #if use_dropout:
  #  x = Dropout(dropout)(x)

  x = GlobalAveragePooling2D()(x)

  x = Dense(n_classes, activation='softmax')(x)
  outputs = x

  model = Model(inputs=inputs, outputs=outputs)

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)

  def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())

  def top_k_categorical_accuracy(y_true, y_pred, k=2):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

  model.compile(optimizer=adam, loss='categorical_crossentropy',
                metrics=['accuracy', categorical_accuracy, top_k_categorical_accuracy])
  model.summary()
  return model


# ______________________________________________________________________________
# Save/Load models

import h5py
import json

def save_my_model(model, name='model'):
  # Store model to file
  #model.summary()
  model.save(name + '.h5')
  model.save_weights(name + '_weights.h5')
  # Store model to json
  with open(name + '.json', 'w') as outfile:
    outfile.write(model.to_json())
  return

def load_my_model(name='model', weights_name='model_weights'):
  with open(name + '.json', 'r') as f:
    json_string = json.dumps(json.load(f))
    model = model_from_json(json_string)
  #model = load_model(name + '.h5')
  model.load_weights(weights_name + '.h5')
  return model
