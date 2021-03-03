import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

import tensorflow as tf

from keras import backend as K

from nn_logging import getLogger
logger = getLogger()


# ______________________________________________________________________________
# The following codes are taken from hls4ml/Javier:
#   https://github.com/hls-fpga-machine-learning/keras-training/blob/muon/train/prune.py

def prune_model(model, percentile=50.):
  sess = K.get_session()

  # Find weights
  allWeights = []
  allWeightsNonRel = []
  allWeightsByLayer = {}
  allWeightsByLayerNonRel = {}
  tensorMaxByLayer = {}
  for layer in model.layers:
    if layer.__class__.__name__ in ['Dense', 'Conv1D']:
      original_w = layer.get_weights()
      weightsByLayer = []
      weightsByLayerNonRel = []
      for my_weights in original_w:
        if len(my_weights.shape) < 2: # bias term, ignore for now
          continue
        #l1norm = tf.norm(my_weights,ord=1)
        elif len(my_weights.shape) == 2: # Dense
          # (n_inputs, n_outputs)
          #tensor_abs = tf.abs(my_weights)
          #tensor_reduce_max_1 = tf.reduce_max(tensor_abs,axis=-1)
          #tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
          tensor_sq = tf.square(my_weights)
          tensor_reduce_sum_1 = tf.reduce_sum(tensor_sq,axis=-1)
          tensor_reduce_sum_2 = tf.reduce_sum(tensor_reduce_sum_1,axis=-1)
        elif len(my_weights.shape) == 3: # Conv1D
          # (filter_width, n_inputs, n_filters)
          #tensor_abs = tf.abs(my_weights)
          #tensor_reduce_max_0 = tf.reduce_max(tensor_abs,axis=-1)
          #tensor_reduce_max_1 = tf.reduce_max(tensor_reduce_max_0,axis=-1)
          #tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
          tensor_sq = tf.square(my_weights)
          tensor_reduce_sum_0 = tf.reduce_sum(tensor_sq,axis=-1)
          tensor_reduce_sum_1 = tf.reduce_sum(tensor_reduce_sum_0,axis=-1)
          tensor_reduce_sum_2 = tf.reduce_sum(tensor_reduce_sum_1,axis=-1)

        #l1norm_val = float(l1norm.eval(session=sess))
        #tensor_max = float(tensor_reduce_max_2.eval(session=sess))
        tensor_max = float(tf.sqrt(tensor_reduce_sum_2).eval(session=sess))
        it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:
          w = it[0]
          allWeights.append(abs(w)/tensor_max)
          allWeightsNonRel.append(abs(w))
          weightsByLayer.append(abs(w)/tensor_max)
          weightsByLayerNonRel.append(abs(w))
          it.iternext()
      if len(weightsByLayer):
        allWeightsByLayer[layer.name] = np.asarray(weightsByLayer)
        allWeightsByLayerNonRel[layer.name] = np.asarray(weightsByLayerNonRel)
        tensorMaxByLayer[layer.name] = tensor_max

  allWeightsArray = np.asarray(allWeights)
  allWeightsArrayNonRel = np.asarray(allWeightsNonRel)

  # Set pruning criteria
  relative_weight_max = np.percentile(allWeightsArray, percentile, axis=-1)

  # Find weights that can be pruned
  weightsPerLayer = {}
  droppedPerLayer = {}
  binaryTensorPerLayer = {}
  for layer in model.layers:
    droppedPerLayer[layer.name] = []
    if layer.__class__.__name__ in ['Dense', 'Conv1D']:
      original_w = layer.get_weights()
      weightsPerLayer[layer.name] = original_w
      for my_weights in original_w:
        if len(my_weights.shape) < 2: # bias term, ignore for now
          continue

        tensor_max = tensorMaxByLayer[layer.name]
        binaryTensorPerLayer[layer.name] = np.ones(my_weights.shape, dtype=np.bool)
        it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:
          w = it[0]
          if abs(w)/tensor_max < relative_weight_max and layer.name.startswith('dense_'):  # only apply to hidden layers
            #print "small relative weight %e/%e = %e -> 0"%(abs(w), tensor_max, abs(w)/tensor_max)
            #w[...] = 0
            droppedPerLayer[layer.name].append((it.multi_index, abs(w)))
            binaryTensorPerLayer[layer.name][it.multi_index] = 0
          it.iternext()
      #print '%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name,layer.count_params())
      #converted_w = convert_kernel(original_w)
      #converted_w = original_w
      #layer.set_weights(converted_w)

  for layer in model.layers:
    logger.info('{0} weights dropped from {1} out of {2} weights.'.format(len(droppedPerLayer[layer.name]), layer.name, layer.count_params()))
  totalDropped = sum(len(k) for v, k in droppedPerLayer.iteritems())
  logger.info('{0} total weights dropped out of {1} total weights.'.format(totalDropped, model.count_params()))
  logger.info('{0} was pruned with {1:.1f}% compression.'.format(model.name, 100.*totalDropped/model.count_params()))
  return weightsPerLayer, droppedPerLayer, binaryTensorPerLayer, allWeightsByLayer, allWeightsArray
