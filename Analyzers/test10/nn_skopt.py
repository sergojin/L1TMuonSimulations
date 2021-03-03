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

from nn_data import muon_data, pileup_data, muon_data_split, pileup_data_split, \
                    mix_training_inputs

from nn_models import create_model, create_model_bn, \
                      create_model_sequential, create_model_sequential_regularized, \
                      lr_decay, modelbestcheck, modelbestcheck_weights

from nn_training import train_model, train_model_sequential


# ______________________________________________________________________________
import skopt
logger.info('Using skopt {0}'.format(skopt.__version__))

use_hpe = ('SLURM_JOB_ID' in os.environ)

if use_hpe:
  infile_muon = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tba.16.npz'
  infile_pileup = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tbd.16.npz'


# ______________________________________________________________________________
# Import muon data
x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = \
    muon_data_split(infile_muon, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, test_size=0.3)

# ______________________________________________________________________________
# Create KerasRegressor

from nn_models import NewKerasRegressor
estimator = NewKerasRegressor(build_fn=create_model_sequential, reg_pt_scale=reg_pt_scale,
                              min_pt=20., max_pt=22., coverage=90.,
                              nvariables=nvariables, lr=learning_rate, l1_reg=l1_reg, l2_reg=l2_reg,
                              epochs=25, batch_size=256, verbose=0)
callbacks_list = [lr_decay,modelbestcheck]

# ______________________________________________________________________________
# Create Objective

from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import mean_squared_error
from skopt.space import Space, Real, Integer, Categorical
from skopt.utils import use_named_args

space = [
  #Real(1e-4, 1e-1, prior='log-uniform', name='lr'),
  Categorical([128, 256, 512, 1024, 2048, 4096, 8192], name='batch_size'),
  Integer(4, 128, name='nodes1'),
  Integer(4, 128, name='nodes2'),
  Integer(4, 128, name='nodes3'),
]

@use_named_args(space)
def objective(**params):
  estimator.set_params(**params)

  #scores = cross_val_score(estimator, x_train, y_train, scoring='neg_mean_squared_error', cv=KFold(3),
  #                         n_jobs=1, verbose=0, fit_params=dict(callbacks=callbacks_list))

  cv_results = cross_validate(estimator, x_train, y_train, scoring={'score': 'neg_mean_squared_error'}, cv=KFold(3),
                              n_jobs=1, verbose=0, fit_params=dict(callbacks=callbacks_list), return_train_score=False)
  scores = cv_results['test_score']

  print("score: {0} +/- {1} with {2}".format(np.mean(scores), np.std(scores), params))
  return -np.mean(scores)


# ______________________________________________________________________________
# Fit

debug = False

if debug:
  estimator.model = estimator.build_fn(**estimator.filter_sk_params(estimator.build_fn))

  from keras.models import Sequential
  fit_args = estimator.filter_sk_params(Sequential.fit)
  print fit_args

  history = estimator.model.fit(x_train, y_train, **fit_args)
  print history.history

  print estimator.score(x_test, y_test)
  print

else:
  #res_gp = skopt.gp_minimize(objective, space, n_calls=50, random_state=0, n_random_starts=12, verbose=True)
  res_gp = skopt.gp_minimize(objective, space, n_calls=1, random_state=0, n_random_starts=1, verbose=True)


# ______________________________________________________________________________
# Results

print("Best score: {0}".format(res_gp.fun))
print("Best parameters: {0!s}".format(res_gp.x))

print("History:")
n_calls = len(res_gp.x_iters)
for i in xrange(n_calls):
  #print res_gp.models[i]
  print(".. {0} x_iter: {1!s} func_val: {2!s}".format(i, res_gp.x_iters[i], res_gp.func_vals[i]))

print("Space: {0!s}".format(res_gp.space))
print("Specs: {0!s}".format(res_gp.specs))

skopt.dump(res_gp, filename='res_gp.pkl', store_objective=False)
res_gp = skopt.load('res_gp.pkl')
print(res_gp)

logger.info('DONE')
