#!/usr/bin/env bash

# Automatically generated on Wed Aug 15 13:41:50 EDT 2018
export HOME=`pwd` 

# The SCRAM architecture and CMSSW version of the submission environment.
readonly SUBMIT_SCRAM_ARCH="slc6_amd64_gcc630"
readonly SUBMIT_CMSSW_VERSION="CMSSW_10_1_7"

# Capture the executable name and job input file from the command line.
readonly CONDOR_EXEC="$(basename $0)"
export CONDOR_EXEC
readonly TARBALL="default.tgz"
readonly ANALYSIS="$1"
readonly JOBID="$2"


echo "$(date) - $CONDOR_EXEC - INFO - condor_scratch: $_CONDOR_SCRATCH_DIR"
echo "$(date) - $CONDOR_EXEC - INFO - pwd: $PWD"
echo "$(date) - $CONDOR_EXEC - INFO - args: $ANALYSIS $JOBID"

echo "$(date) - $CONDOR_EXEC - INFO - Unpacking files"
tar xzf $TARBALL

echo "$(date) - $CONDOR_EXEC - INFO - Setting up $SUBMIT_CMSSW_VERSION"

# Setup the CMS software environment.
export SCRAM_ARCH="$SUBMIT_SCRAM_ARCH"
source /cvmfs/cms.cern.ch/cmsset_default.sh

# Checkout the CMSSW release and set the runtime environment. These
# commands are often invoked by their aliases "cmsrel" and "cmsenv".
#scram project CMSSW "$SUBMIT_CMSSW_VERSION"
cd "$SUBMIT_CMSSW_VERSION/src"
#scramv1 b ProjectRename
eval "$(scramv1 runtime -sh)"

echo $CMSSW_RELEASE_BASE
echo $CMSSW_BASE
which python
echo $HOME
#XDG_CACHE_HOME=$_CONDOR_SCRATCH_DIR/.cache

echo "$(date) - $CONDOR_EXEC - INFO - Setting up virtualenv"
#source venv/bin/activate

# Change back to the worker node's scratch directory.
cd "$_CONDOR_SCRATCH_DIR"

echo "$(date) - $CONDOR_EXEC - INFO - pwd: $PWD"

echo "$(date) - $CONDOR_EXEC - INFO - ls: -"
ls -a

# Do Science
echo "$(date) - $CONDOR_EXEC - INFO - Stand back I'm going to try Science!"

echo "python rootpy_trackbuilding7.py $ANALYSIS $JOBID"

python rootpy_trackbuilding7.py $ANALYSIS $JOBID

EXIT_STATUS=$?
ERROR_TYPE=""
ERROR_MESSAGE="This is an error message."

echo "$(date) - $CONDOR_EXEC - INFO - Postprocessing"

# Rename output files
[ -f histos_tba.root ] && mv histos_tba.root histos_tba_$JOBID.root
[ -f histos_tba.npz  ] && mv histos_tba.npz  histos_tba_$JOBID.npz
[ -f histos_tbb.root ] && mv histos_tbb.root histos_tbb_$JOBID.root
[ -f histos_tbb.npz  ] && mv histos_tbb.npz  histos_tbb_$JOBID.npz
[ -f histos_tbc.root ] && mv histos_tbc.root histos_tbc_$JOBID.root
[ -f histos_tbc.npz  ] && mv histos_tbc.npz  histos_tbc_$JOBID.npz
[ -f histos_tbd.root ] && mv histos_tbd.root histos_tbd_$JOBID.root
[ -f histos_tbd.npz  ] && mv histos_tbd.npz  histos_tbd_$JOBID.npz

# Prepare reports
if [ $EXIT_STATUS -ne 0 ]; then
  cat << EOF > FrameworkJobReport.xml
<FrameworkJobReport>
<FrameworkError ExitStatus="$EXIT_STATUS" Type="$ERROR_TYPE" >
$ERROR_MESSAGE
</FrameworkError>
</FrameworkJobReport>
EOF
fi

# Clean up
tar tzf $TARBALL | xargs rm -rf
rm -rf $TARBALL
rm -rf *.pyc

echo "$(date) - $CONDOR_EXEC - INFO - ls: -"
ls -l

echo "$(date) - $CONDOR_EXEC - INFO - Science complete!"
exit $EXIT_STATUS