from nn_logging import getLogger
logger = getLogger()

# ______________________________________________________________________________
# Globals

superstrip_size = 16

n_zones = 7

rows_per_zone = 9

n_rows = n_zones * rows_per_zone

n_columns = 5040 // superstrip_size

n_channels = 2

n_classes = 21

dropout = 0.2

learning_rate = 1e-3

gradient_clip_norm = 100.

infile_images = '../test7/histos_tbe.18.npz'


# ______________________________________________________________________________
# Import all the libs
import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
OLD_STDOUT = sys.stdout

logger.info('Using cmssw {0}'.format(os.environ['CMSSW_VERSION']))

import numpy as np
np.random.seed(2023)
logger.info('Using numpy {0}'.format(np.__version__))

import tensorflow as tf
logger.info('Using tensorflow {0}'.format(tf.__version__))

import keras
from keras import backend as K
#K.set_epsilon(1e-08)
#K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True)))
logger.info('Using keras {0}'.format(keras.__version__))
logger.info('.. list devices: {0}'.format(K.get_session().list_devices()))

import scipy
logger.info('Using scipy {0}'.format(scipy.__version__))

import sklearn
logger.info('Using sklearn {0}'.format(sklearn.__version__))

import matplotlib.pyplot as plt
#from matplotlib import colors
#%matplotlib inline

