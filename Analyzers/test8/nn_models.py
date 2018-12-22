import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model, clone_model, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, Concatenate, Lambda, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from keras.regularizers import Regularizer
from keras.constraints import Constraint
from keras import initializers, regularizers, optimizers, losses

import h5py
import json
import functools


# ______________________________________________________________________________
# New leaky relu
def NewLeakyReLU(x, alpha=0., max_value=None):
  return K.relu(x, alpha=alpha, max_value=max_value)

# ______________________________________________________________________________
# New tanh
def NewTanh(x):
  return K.tanh(x)
  #return 1.7159 * K.tanh(x * 2./3.)
  #return K.clip(x, -1., 1.)

# ______________________________________________________________________________
# New elu
def NewElu(x, alpha=1.0):
  return K.elu(x, alpha) + alpha*1.0 + 1e-15

# ______________________________________________________________________________
def count_params(nvariables=80, nodes1=64, nodes2=32, nodes3=16, npredictions=1, use_bn=False):
  n = 0
  if use_bn:
    # Hidden layer 1
    n += (nvariables * nodes1 + 4*nodes1)
    # Hidden layer 2
    n += (nodes1 * nodes2 + 4*nodes2)
    # Hidden layer 3
    n += (nodes2 * nodes3 + 4*nodes3)
    # Output layer
    n += (nodes3 * 1 + 1) * npredictions
  else:
    # Hidden layer 1
    n += (nvariables * nodes1 + nodes1)
    # Hidden layer 2
    n += (nodes1 * nodes2 + nodes2)
    # Hidden layer 3
    n += (nodes2 * nodes3 + nodes3)
    # Output layer
    n += (nodes3 * 1 + 1) * npredictions
  return n

class LCountParams(Regularizer):
  """Regularizer that penalizes large number of parameters.
  Copied from class L1L2 from https://github.com/keras-team/keras/blob/master/keras/regularizers.py

  # Arguments
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  """

  def __init__(self, l1=0., l2=0.):
    self.l1 = K.cast_to_floatx(l1)
    self.l2 = K.cast_to_floatx(l2)

  def __call__(self, x):
    regularization = 0.
    if self.l1:
      regularization += self.l1 * K.abs(K.cast_to_floatx(K.count_params(x)))
    if self.l2:
      regularization += self.l2 * K.square(K.cast_to_floatx(K.count_params(x)))
    return regularization

  def get_config(self):
    return {'l1': float(self.l1),
            'l2': float(self.l2)}

# ______________________________________________________________________________
class ZeroSomeWeights(Constraint):
  """ZeroSomeWeights weight constraint.
  Constrains certain weights incident to each hidden unit
  to be zero.
  Copied from https://github.com/hls-fpga-machine-learning/keras-training/blob/muon/models/constraints.py

  # Arguments
      binary_tensor: binary tensor of 0 or 1s corresponding to which weights to zero.
  """

  def __init__(self, binary_tensor=None):
    self.binary_tensor = binary_tensor

  def __call__(self, w):
    if self.binary_tensor is not None:
      w *= K.cast(self.binary_tensor, K.floatx())
    return w

  def get_config(self):
    return {'binary_tensor': self.binary_tensor.tolist()}

# ______________________________________________________________________________
# Huber loss

def huber_loss(y_true, y_pred, delta=1.345):
  x = K.abs(y_true - y_pred)
  squared_loss = 0.5*K.square(x)
  absolute_loss = delta * (x - 0.5*delta)
  #xx = K.switch(x < delta, squared_loss, absolute_loss)
  xx = tf.where(x < delta, squared_loss, absolute_loss)  # needed for tensorflow
  return K.mean(xx, axis=-1)

def masked_huber_loss(y_true, y_pred, delta=1.345, mask_value=100.):
  x = K.abs(y_true - y_pred)
  squared_loss = 0.5*K.square(x)
  absolute_loss = delta * (x - 0.5*delta)
  #xx = K.switch(x < delta, squared_loss, absolute_loss)
  xx = tf.where(x < delta, squared_loss, absolute_loss)  # needed for tensorflow

  mask = K.not_equal(y_true, mask_value)
  mask = K.cast(mask, K.floatx())
  xx *= mask
  xx /= (K.mean(mask) + K.epsilon())
  return K.mean(xx, axis=-1)

def unmasked_huber_loss(y_true, y_pred, delta=1.345, mask_value=100.):
  x = K.abs(y_true - y_pred)
  squared_loss = 0.5*K.square(x)
  absolute_loss = delta * (x - 0.5*delta)
  #xx = K.switch(x < delta, squared_loss, absolute_loss)
  xx = tf.where(x < delta, squared_loss, absolute_loss)  # needed for tensorflow

  mask = K.not_equal(y_true, mask_value)
  x_pu = (K.abs(y_pred)-100./10.)
  log_loss = K.log(1+K.exp(-x_pu))
  log_loss *= 20.
  xx = tf.where(mask, xx, log_loss)
  return K.mean(xx, axis=-1)

# ______________________________________________________________________________
# Binary crossentropy

# See: https://github.com/keras-team/keras/blob/master/keras/losses.py
#def binary_crossentropy(y_true, y_pred):
#  return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

# See: https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
#def binary_crossentropy(target, output, from_logits=False):
#    # Note: tf.nn.sigmoid_cross_entropy_with_logits
#    # expects logits, Keras expects probabilities.
#    if not from_logits:
#        # transform back to logits
#        _epsilon = tf.convert_to_tensor(epsilon(), dtype=output.dtype.base_dtype)
#        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
#        output = tf.log(output / (1 - output))
#    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

def weighted_crossentropy(target, output, pos_weight, from_logits=False):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
      # transform back to logits
      _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
      output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
      output = tf.log(output / (1 - output))
    return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=pos_weight)

def masked_binary_crossentropy(y_true, y_pred, mask_value=100., pos_weight=0.333333):
  #xx = K.binary_crossentropy(y_true, y_pred)
  xx = weighted_crossentropy(y_true, y_pred, pos_weight=pos_weight) * 2/(1+pos_weight)

  mask = K.not_equal(y_true, mask_value)
  mask = K.cast(mask, K.floatx())
  xx *= mask
  xx /= (K.mean(mask) + K.epsilon())
  return K.mean(xx, axis=-1)

# ______________________________________________________________________________
# Callbacks

def lr_schedule(epoch, lr):
  if (epoch % 10) == 0 and epoch != 0:
    lr *= 0.9
  return lr

lr_decay = LearningRateScheduler(lr_schedule, verbose=0)

terminate_on_nan = TerminateOnNaN()

modelbestcheck = ModelCheckpoint(filepath='model_bchk.h5', monitor='val_loss', verbose=1, save_best_only=True)
modelbestcheck_weights = ModelCheckpoint(filepath='model_bchk_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

# ______________________________________________________________________________
# Custom objects

def update_keras_custom_objects():
  custom_objects = {
    'masked_huber_loss': masked_huber_loss,
    'masked_binary_crossentropy': masked_binary_crossentropy,
    'NewLeakyReLU': NewLeakyReLU,
    'NewTanh': NewTanh,
    'NewElu': NewElu,
    'LCountParams': LCountParams,
    'ZeroSomeWeights': ZeroSomeWeights,
  }

  from keras.utils.generic_utils import get_custom_objects
  get_custom_objects().update(custom_objects)

# ______________________________________________________________________________
# Create models
def create_model(nvariables, lr=0.001, clipnorm=10., nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0, l1_reg=0.0, l2_reg=0.0):
  regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer)(inputs)
  if nodes2:
    x = Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer)(x)
    if nodes3:
      x = Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer)(x)

  # Output nodes
  regr = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='regr')(x)
  discr = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='discr')(x)

  # Create model
  model = Model(inputs=inputs, outputs=[regr, discr])

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)
  model.compile(optimizer=adam,
    loss={'regr': masked_huber_loss, 'discr': masked_binary_crossentropy},
    loss_weights={'regr': 1.0, 'discr': discr_loss_weight},
    #metrics={'regr': ['acc', 'mse', 'mae'], 'discr': ['acc',]}
    )
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_bn(nvariables, lr=0.001, clipnorm=10., nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0,
                    l1_reg=0.0, l2_reg=0.0, use_bn=True, use_dropout=False):
  regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
  inputs = Input(shape=(nvariables,), dtype='float32')
  if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(inputs)  

  x = Dense(nodes1, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False)(x)
  if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
  x = Activation('tanh')(x)
  if use_dropout: x = Dropout(0.2)(x)
  if nodes2:
    x = Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False)(x)
    if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
    x = Activation('tanh')(x)
    if use_dropout: x = Dropout(0.2)(x)
    if nodes3:
      x = Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False)(x)
      if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
      x = Activation('tanh')(x)
      if use_dropout: x = Dropout(0.2)(x)

  # Output nodes
  regr = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='regr')(x)
  discr = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='discr')(x)

  # Create model
  model = Model(inputs=inputs, outputs=[regr, discr])

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)
  model.compile(optimizer=adam,
    loss={'regr': masked_huber_loss, 'discr': masked_binary_crossentropy},
    loss_weights={'regr': 1.0/discr_loss_weight, 'discr': 1.0},
    #metrics={'regr': ['acc', 'mse', 'mae'], 'discr': ['acc',]}
    )
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_pruned(nvariables, lr=0.001, clipnorm=10., nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0,
                        l1_reg=0.0, l2_reg=0.0, use_bn=True, use_dropout=False,
                        constraint1=None, constraint2=None, constraint3=None):
  regularizer = None  # disable
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint1, use_bias=False)(inputs)
  if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
  x = Activation('tanh')(x)
  if use_dropout: x = Dropout(0.2)(x)
  if nodes2:
    x = Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint2, use_bias=False)(x)
    if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
    x = Activation('tanh')(x)
    if use_dropout: x = Dropout(0.2)(x)
    if nodes3:
      x = Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint3, use_bias=False)(x)
      if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
      x = Activation('tanh')(x)
      if use_dropout: x = Dropout(0.2)(x)

  # Output nodes
  regr = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='regr')(x)
  discr = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='discr')(x)

  # Create model
  model = Model(inputs=inputs, outputs=[regr, discr])

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)
  model.compile(optimizer=adam,
    loss={'regr': masked_huber_loss, 'discr': masked_binary_crossentropy},
    loss_weights={'regr': 1.0, 'discr': discr_loss_weight},
    #metrics={'regr': ['acc', 'mse', 'mae'], 'discr': ['acc',]}
    )
  model.summary()
  return model

# ______________________________________________________________________________
def log_prob_normal(x, loc=0.0, scale=1.0):
  y = (x - loc) / scale
  y = -0.5*K.square(y) - 0.5*K.log(2*np.pi) - K.log(scale)
  return y

def log_prob_softmax(x):
  return tf.nn.log_softmax(K.clip(x,K.epsilon(),1.0))

def log_prob(x, mus, sigmas, pi):
  distribution_log_probs = log_prob_normal(x, mus, sigmas)
  #cat_log_probs = log_prob_softmax(pi)
  cat_log_probs = K.log(pi)
  final_log_probs = tf.add(distribution_log_probs, cat_log_probs)
  result = tf.reduce_logsumexp(final_log_probs, axis=-1)
  return result

def mixture_loss(y_true, y_pred, mus, sigmas, pi):
  loss = -log_prob(y_true, mus, sigmas, pi)
  return K.mean(loss, axis=-1)

def mixture_loss_for_keras(mus, sigmas, pi):
  @functools.wraps(mixture_loss)
  def loss(y_true, y_pred):
    return mixture_loss(y_true, y_pred, mus, sigmas, pi)
  return loss

def create_model_mdn(nvariables, lr=0.001, clipnorm=10., nodes1=64, nodes2=32, nodes3=16, mixture=16, discr_loss_weight=1.0,
                     l1_reg=0.0, l2_reg=0.0, use_bn=True, use_dropout=False,
                     constraint1=None, constraint2=None, constraint3=None):
  regularizer = None  # disable
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint1, use_bias=False)(inputs)
  if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
  x = Activation('tanh')(x)
  if use_dropout: x = Dropout(0.2)(x)
  if nodes2:
    x = Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint2, use_bias=False)(x)
    if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
    x = Activation('tanh')(x)
    if use_dropout: x = Dropout(0.2)(x)
    if nodes3:
      x = Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint3, use_bias=False)(x)
      if use_bn: x = BatchNormalization(epsilon=1e-4, momentum=0.9)(x)
      x = Activation('tanh')(x)
      if use_dropout: x = Dropout(0.2)(x)

  # Output nodes
  mus = Dense(mixture, activation=None, kernel_initializer='glorot_uniform', name='mus')(x)  # the means
  sigmas = Dense(mixture, activation=NewElu, kernel_initializer='glorot_uniform', name='sigmas')(x)  # the variance
  pi = Dense(mixture, activation='softmax', kernel_initializer='glorot_uniform', name='pi')(x)  # the mixture components
  outputs = Concatenate(axis=1)([pi, mus, sigmas])

  # Create model
  model = Model(inputs=inputs, outputs=outputs)

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)
  keras_loss = mixture_loss_for_keras(mus=mus, sigmas=sigmas, pi=pi)
  model.compile(optimizer=adam, loss=keras_loss)
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_sequential(nvariables, lr=0.001, clipnorm=10., nodes1=64, nodes2=32, nodes3=16, l1_reg=0.0, l2_reg=0.0):
  regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)

  model = Sequential()
  model.add(Dense(nodes1, input_shape=(nvariables,), activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
  #model.add(Dropout(0.2))
  if nodes2:
    model.add(Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
    #model.add(Dropout(0.2))
    if nodes3:
      model.add(Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
      #model.add(Dropout(0.2))

  # Output node
  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)
  model.compile(optimizer=adam, loss=huber_loss, metrics=['acc'])
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_sequential_bn(nvariables, lr=0.001, clipnorm=10., nodes1=64, nodes2=32, nodes3=16,
                               l1_reg=0.0, l2_reg=0.0, use_bn=True, use_dropout=False):
  regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)

  model = Sequential()
  model.add(Dense(nodes1, input_shape=(nvariables,), kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False))
  if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
  model.add(Activation('tanh'))
  if use_dropout: model.add(Dropout(0.2))
  if nodes2:
    model.add(Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False))
    if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
    model.add(Activation('tanh'))
    if use_dropout: model.add(Dropout(0.2))
    if nodes3:
      model.add(Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False))
      if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
      model.add(Activation('tanh'))
      if use_dropout: model.add(Dropout(0.2))

  # Output node
  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr, clipnorm=clipnorm)
  model.compile(optimizer=adam, loss=huber_loss, metrics=['acc'])
  #model.compile(optimizer=adam, loss=unmasked_huber_loss, metrics=['acc'])
  model.summary()
  return model

# ______________________________________________________________________________
# Save/Load models
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


# ______________________________________________________________________________
# Scoring for cross-validation
# Based on https://github.com/keras-team/keras/blob/master/keras/wrappers/scikit_learn.py

from keras.wrappers.scikit_learn import KerasRegressor

class NewKerasRegressor(KerasRegressor):
  """KerasRegressor with custom 'score' function
  """

  def __init__(self, build_fn=None, **sk_params):
    super(NewKerasRegressor, self).__init__(build_fn=build_fn, **sk_params)

  def fit(self, x, y, **kwargs):
    print('.. fitting x (shape:{0}) & y (shape:{1}) with params: {2} & {3}'.format(x.shape, y.shape, self.filter_sk_params(self.build_fn), kwargs))
    return super(NewKerasRegressor, self).fit(x, y, **kwargs)
