#!/bin/bash
# syntax: either 
#     run_MP.sh /path/to/folder config.yaml
# or
#     run_MP.sh /path/to/folder checkpoint_name
#
# script checks for a yaml extension an assumes intent based on this. 

if [ -z "$1" ]
then
    echo "Missing an experiment directory"
    exit 1
fi

logDir=$1
mkdir -p $logDir
export PYTHONHASHSEED=0
DATETIME=`date +"%Y-%m-%d_%T"`

if [[ $2 == *.yaml ]]
then
  # No checkpoint, start a new run
  config_file=$2
  base_name=$(basename ${config_file})
  cp $config_file $logDir/$base_name

  echo "Starting new MarcoPolo run in $logDir from $config_file"
  python -u master.py \
    --log_file $logDir \
    --config $config_file \
    2>&1 | tee $logDir/run.$DATETIME.log 
else
  echo Restarting from checkpoint
  chkpt=$2
  if [ -d $logDir/Checkpoints/$chkpt ] 
  then
    # A checkpoint has been provided, load from that.
    echo "Continuing MarcoPolo run in $logDir at checkpoint $chkpt"
    echo "" 
    python -u master.py \
      --log_file $logDir \
      --start_from $chkpt \
      2>&1 | tee $logDir/run.$chkpt.$DATETIME.log 
  else 
    echo "Checkpoint $logDir/Checkpoints/$chkpt does not exist"
  fi
fi