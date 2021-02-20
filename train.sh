#!/bin/bash

#python tools/banet_train.py --epoch-to-train 150 --name NAME --finetune-from ./res/NAME_OF_WEIGHTS.pth

for (( i=1; i<=406; i=i+1 ))
do
  echo 'i is '${i}
  echo '======='
  ## given the argument ''--epoch-to-train'' value '1', model will do the evaluation every epoch
  python tools/banet_train.py --epoch-to-train 1 --name NAME --finetune-from ./res/NAME_OF_WEIGHTS.pth
done
