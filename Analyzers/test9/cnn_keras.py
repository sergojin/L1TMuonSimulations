# Based on official TensorFlow implementation of MNIST
# https://github.com/tensorflow/models/tree/master/official/mnist

import numpy as np
import tensorflow as tf

from cnn_globals import (superstrip_size, n_zones, rows_per_zone, n_rows, n_columns, n_channels, n_classes, dropout, learning_rate, gradient_clip_norm)

from cnn_models import create_model, save_my_model


# ______________________________________________________________________________
def parse_image_fn(pixels, channels):
  n = pixels.shape[0]
  assert(pixels.shape == (n,2))
  #assert(channels.shape == (n,n_channels))

  bad_pixel = -99
  mask = (pixels[:,0] != bad_pixel)

  image_shape = (n_rows, n_columns, n_channels)
  image = np.zeros(image_shape, dtype=channels.dtype)
  #image[pixels[mask,0], pixels[mask,1]] = channels[mask]
  image[pixels[mask,0], pixels[mask,1]] = np.column_stack((channels[mask,0],1-channels[mask,0]))
  return image

def parse_label_fn(labels):
  assert(labels.shape == (3,))
  lb = labels[:1]
  lb = tf.keras.utils.to_categorical(lb[0], num_classes=n_classes)  # in Keras, use one-hot encoding
  return lb


# ______________________________________________________________________________
def define_reiam_flags():
  """Define flags which will be used for MNIST models."""
  from cnn_utils import define_reiam_base_flags, clear_flags, unparse_flags, set_defaults

  define_reiam_base_flags()
  set_defaults(batch_size=100,
               num_epochs=5,
               data_dir='./reiam_data',
               model_dir='./reiam_model',
               benchmark_logger_type='BenchmarkFileLogger',
               benchmark_log_dir='./reiam_model',
               hooks='LoggingTensorHook,ProfilerHook,ExamplesPerSecondHook,LoggingMetricHook')


# ______________________________________________________________________________
def run_reiam(flags_obj, data):
  """Run MNIST training and eval loop.
  Args:
    flags_obj: An object containing parsed flag values.
  """

  assert(tf.keras.backend.backend() == 'tensorflow')
  assert(tf.keras.backend.image_data_format() == 'channels_last')

  (images_px_train, images_px_test, images_ch_train, images_ch_test, labels_train, labels_test, parameters_train, parameters_test) = data

  images = map(lambda (px, ch): parse_image_fn(px, ch), zip(images_px_train, images_ch_train))
  labels = map(lambda lb: parse_label_fn(lb), labels_train)
  images = np.asarray(images)
  labels = np.asarray(labels)

  params = {
      #'data_format': data_format,
      #'multi_gpu': multi_gpu,
      'n_rows': n_rows,
      'n_columns': n_columns,
      'n_channels': n_channels,
      'n_classes': n_classes,
      'dropout': dropout,
      'learning_rate': learning_rate,
      'gradient_clip_norm': gradient_clip_norm,
  }

  model = create_model(params, use_cnn_keras=True)

  # ____________________________________________________________________________
  # Train model
  import datetime
  from nn_logging import getLogger
  from nn_training import TrainingLog

  start_time = datetime.datetime.now()
  logger = getLogger()
  logger.info('Begin training ...')

  with TrainingLog() as tlog:  # redirect sys.stdout
    history = model.fit(images, labels, epochs=flags_obj.num_epochs, batch_size=flags_obj.batch_size,
                        validation_split=0.1, verbose=1)

  logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))

  save_my_model(model, name='model_cnn')

  # ____________________________________________________________________________
  # Evaluate model
  #images = map(lambda (px, ch): parse_image_fn(px, ch), zip(images_px_test, images_ch_test))
  #labels = map(lambda lb: parse_label_fn(lb), labels_test)
  #images = np.asarray(images)
  #labels = np.asarray(labels)
  #
  #loss_and_metrics = model.evaluate(images, labels, batch_size=flags_obj.batch_size, verbose=0)
  #logger.info('loss and metrics: {0}'.format(loss_and_metrics))

  return model
