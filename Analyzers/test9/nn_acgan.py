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

def masked_binary_crossentropy(y_true, y_pred, mask_value=100.):
  xx = K.binary_crossentropy(y_true, y_pred)

  mask = K.not_equal(y_true, mask_value)
  mask = K.cast(mask, K.floatx())
  xx *= mask
  xx /= (K.mean(mask) + K.epsilon())
  return K.mean(xx, axis=-1)


from six.moves import range, zip

def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]

from nn_globals import (learning_rate, gradient_clip_norm, reg_pt_scale, discr_loss_weight)

from nn_encode import nlayers, nvariables


# ______________________________________________________________________________
# Codes from https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py
#            https://github.com/eriklindernoren/Keras-GAN/blob/master/sgan/sgan.py

class ACGAN():
    def __init__(self):
        # Set loss and optimizers
        #adam = optimizers.Adam(lr=learning_rate, clipnorm=gradient_clip_norm)
        adam = optimizers.Adam(0.001, 0.5)  #FIXME
        losses = [masked_huber_loss, masked_binary_crossentropy]
        discr_loss_weight = 10.0  #FIXME
        loss_weights = [1.0, discr_loss_weight]

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses[1],
            optimizer=adam,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(nvariables,), dtype='float32')
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # of that image
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(noise, [img, valid])
        self.combined.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=adam)

    def build_generator(self,
                        nodes1=50, nodes2=30, nodes3=20,
                        l1_reg=0.0, l2_reg=0.0, use_bn=True, use_dropout=False):
        regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)

        model = Sequential()
        model.add(Dense(nodes1, input_shape=(nvariables,), kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=(not use_bn)))
        if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
        model.add(Activation('tanh'))
        if use_dropout: model.add(Dropout(0.2))
        if nodes2:
          model.add(Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=(not use_bn)))
          if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
          model.add(Activation('tanh'))
          if use_dropout: model.add(Dropout(0.2))
          if nodes3:
            model.add(Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=(not use_bn)))
            if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
            model.add(Activation('tanh'))
            if use_dropout: model.add(Dropout(0.2))

        model.summary()

        noise = Input(shape=(nvariables,), dtype='float32')
        features = model(noise)

        # Output node
        img = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='regr')(features)
        return Model(noise, img)

    def build_discriminator(self,
                            nodes1=20, nodes2=20, nodes3=20,
                            l1_reg=0.0, l2_reg=0.0, use_bn=False, use_dropout=False):
        regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)

        model = Sequential()
        model.add(Dense(nodes1, input_shape=(1,), kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=(not use_bn)))
        if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
        #model.add(Activation('tanh'))
        model.add(LeakyReLU(alpha=0.2))
        if use_dropout: model.add(Dropout(0.2))
        if nodes2:
          model.add(Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=(not use_bn)))
          if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
          #model.add(Activation('tanh'))
          model.add(LeakyReLU(alpha=0.2))
          if use_dropout: model.add(Dropout(0.2))
          if nodes3:
            model.add(Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=(not use_bn)))
            if use_bn: model.add(BatchNormalization(epsilon=1e-4, momentum=0.9))
            #model.add(Activation('tanh'))
            model.add(LeakyReLU(alpha=0.2))
            if use_dropout: model.add(Dropout(0.2))

        model.summary()

        img = Input(shape=(1,), dtype='float32')

        #features = model(img)

        features = Lambda(lambda x: K.abs(x))(img)  # take absolute value
        #features = Lambda(lambda x: K.log(1+K.exp(-(K.abs(100./x)-100./14.)/0.2)))(img)  # convert to logit
        features = model(features)

        # Output node
        valid = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='discr')(features)
        return Model(img, valid)

    def pre_train(self, x_train, y_train):
        mask_value = 100
        y_train_new = [y_train[0], np.full((y_train[0].shape[0],), mask_value, dtype=np.float32)]

        #epochs = 200
        epochs = 20
        batch_size = 256*4*4

        history = self.combined.fit(x_train, y_train_new,
                                    epochs=epochs, batch_size=batch_size,
                                    validation_split=0.1, verbose=1)

        self.save_model()
        return

    def train(self, x_train, y_train):
        #epochs = 200
        epochs = 20
        batch_size = 256*4*4

        num_train_samples = x_train.shape[0]
        index_array = np.arange(num_train_samples)

        # Loop over epochs
        for epoch in range(epochs):
            # Shuffle
            np.random.shuffle(index_array)

            # Loop over batches
            batches = make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
              batch_ids = index_array[batch_start:batch_end]

              # Select input
              noise = x_train[batch_ids]

              # Generate new images
              imgs = self.generator.predict_on_batch(noise)

              # Image labels.
              img_labels = y_train[0][batch_ids]
              valid = y_train[1][batch_ids]

              # Train the discriminator
              d_loss = self.discriminator.train_on_batch(imgs, valid)

              # Train the generator
              # (with fewer updates)
              if batch_index == len(batches)-1:
                g_loss = self.combined.train_on_batch(noise, [img_labels, valid])
              continue  # end loop over batches

            # Plot the progress
            #print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            print ("%d [D loss: %r] [G loss: %r]" % (epoch, d_loss, g_loss))

            # If at save interval => save generated image samples
            #if epoch % sample_interval == 0:
            #    self.save_model()
            #    self.sample_images(epoch)
            continue  # end loop over epochs

        self.save_model()
        return

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
        save(self.combined, "adversarial")
