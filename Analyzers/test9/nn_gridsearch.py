"""
export SCRAM_ARCH=slc6_amd64_gcc630
cmsrel CMSSW_10_1_7
cd CMSSW_10_1_7/src
cmsenv
virtualenv venv
source venv/bin/activate
pip install -U pip
pip install scikit-optimize
cd ../..

# More info about sklearn/skopt
#   http://scikit-learn.org/stable/modules/cross_validation.html
#   http://scikit-learn.org/stable/modules/grid_search.html
#   https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/sklearn-gridsearchcv-replacement.ipynb
"""


# ______________________________________________________________________________
from nn_globals import *

from nn_encode import nlayers, nvariables

from nn_data import (muon_data_split, pileup_data_split, mix_training_inputs)

from nn_models import (create_model, create_model_bn, create_model_pruned,
                       create_model_sequential, create_model_sequential_bn,
                       lr_decay, modelbestcheck, modelbestcheck_weights)

from nn_training import train_model

from nn_pruning import prune_model


# ______________________________________________________________________________
import skopt
logger.info('Using skopt {0}'.format(skopt.__version__))

use_hpe = ('SLURM_JOB_ID' in os.environ)

if use_hpe:
  infile_muon = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tba.18.npz'
  infile_pileup = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tbd.18.npz'


# ______________________________________________________________________________
# Import muon data
# 'x' is the input variables with shape (n, 87), 'y' is the q/pT with shape (n, 1)
#x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = \
#    muon_data_split(infile_muon, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, test_size=0.31)

# Use ShuffleSplit as the CV iterator
from nn_data import muon_data
from sklearn.model_selection import ShuffleSplit
x, y, w, x_mask = muon_data(infile_muon, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, correct_for_eta=False)
cv = ShuffleSplit(n_splits=1, test_size=0.31)

# ______________________________________________________________________________
# Create KerasRegressor

from nn_models import NewKerasRegressor
estimator = NewKerasRegressor(build_fn=create_model_sequential_bn,
                              nvariables=nvariables, lr=learning_rate, clipnorm=gradient_clip_norm, l1_reg=l1_reg, l2_reg=l2_reg,
                              nodes1=40, nodes2=30, nodes3=20,
                              epochs=100, batch_size=4096, verbose=0)
callbacks_list = [lr_decay,modelbestcheck]


# ______________________________________________________________________________
# Create GridSearchCV

from sklearn.model_selection import cross_val_score, cross_validate, KFold, GridSearchCV

nodes1 = [30,40,60,80]
nodes2 = [20,30,40]
nodes3 = [10,20,30]
lr = [0.001, 0.01]
batches = [256, 512, 1024, 4096]
param_grid = dict(lr=lr, nodes1=nodes1, nodes2=nodes2, nodes3=nodes3)
#param_grid = dict(lr=lr, batch_size=batches)
logger.info('Using parameter grid: %r' % param_grid)

opt = GridSearchCV(estimator=estimator, param_grid=param_grid,
                   n_jobs=1, cv=cv, verbose=0, refit=False, return_train_score=False)

logger.info('Begin training ...')
opt.fit(x, y)


# ______________________________________________________________________________
# Create BayesSearchCV

#from sklearn.model_selection import cross_val_score, cross_validate, KFold
#from skopt.space import Space, Real, Integer, Categorical
#from skopt import BayesSearchCV
#search_spaces = {
#  #'lr': Real(1e-4, 1e-2, prior='log-uniform'),  # 'lr': Real(1e-4, 2e-3),
#  'batch_size': Categorical([128, 256, 512, 1024, 2048, 4096, 8192]),
#  #'nodes1': Integer(4, 256),
#  #'nodes2': Integer(4, 256),
#  #'nodes3': Integer(4, 256),
#}

##opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, scoring='neg_mean_squared_error', fit_params=dict(callbacks=callbacks_list))
#opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, scoring=None, fit_params=dict(callbacks=callbacks_list))
##opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, n_points=1, cv=KFold(5), n_jobs=5, scoring=None, return_train_score=True)
#opt.fit(x_train, y_train)


# ______________________________________________________________________________
# Results

print('Best: %f using %s' % (opt.best_score_, opt.best_params_))
#opt.best_estimator_.model.summary()
means = opt.cv_results_['mean_test_score']
stds = opt.cv_results_['std_test_score']
params = opt.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print('%f (+/-%f) with: %r' % (mean, stdev, param))
print

print(opt)
for k, v in opt.cv_results_.iteritems():
  print("'%s': %r" % (k, v))

# Persistency
from sklearn.externals import joblib
joblib.dump(opt, 'dump.pkl')
#loaded_opt = joblib.load('dump.pkl')

logger.info('DONE')
