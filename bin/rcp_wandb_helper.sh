#!/bin/bash

PATH=$PATH:$HOME/minimamba/bin
cd $1 

source .venv/bin/activate
cd $2
$1/scripts/rcp_wandb agent $3
