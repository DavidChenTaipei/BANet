#!/bin/bash
for (( i=1; i<=1; i=i+1 ))
do
  echo 'i is '${i}
  echo '======='
  python tools/banet_train.py --epoch-to-train 150 --name banet --finetune-from ./res/banet.pth
done
