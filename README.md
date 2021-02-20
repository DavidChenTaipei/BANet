# BANet
### Bilateral attention decoder: A lightweight decoder for real-time semantic segmentation

This is my implement code of the [paper](https://doi.org/10.1016/j.neunet.2021.01.021): **Bilateral attention decoder: A lightweight decoder for real-time semantic segmentation** and the sturcture of the code is based on the implementation of [BiSeNet](https://github.com/CoinCheung/BiSeNet) by CoinCheung.

The structure and all the configs are set as what author mentions in the paper.

## TODO : 
There is a bug while training with the warm up learning rate policy, the loss of the model will turn into **nan** , I will fix the problem asap. For now, you can train with the model through the script **train.sh** , however, after the warming up epoch, please train the model with the for loop in **train.sh** and change the [code](https://github.com/DavidChenTaipei/BANet/blob/1206c8cde021cfcc38ac14b0d30b62940a545fdf/lib/lr_scheduler.py#L34) in **lr_schedulor.py** to train without warm up policy. The loss will not turn into **nan** without warm up.

## Introduction
Now, I'm going to do the brief introduction to each file
### file : best_miou.txt
The best miou during training will be record here and the weight of it will be stored with **NAME_best.pth**.

### file: lr_record.txt
The learning rate need to be set at here, and will be changed due to the learning rate policy.
NOTE THAT, the learning rate here will ovewrite the learning rate in config file.

### folder : Configs 
The file config.py is the file where the parameters are set.

### folder : datasets
Please put the dataset in the subfolder cityscapes of this folder, and check the relative path to the data in those two text file

### folder : lib
In the lib folder, the **base_dataset.py** and **cityscapes_cv2.py** is where the dataset to be prepared. After these, the file **logger.py** is to set the information during training and **lr_scheduler.py** and **meters.py** are for learning rate policy and to set the meters for traininf repectively.Moreover, there is a subfolder named **model** and the struture of the model is written here. 

### folder : res
The model will store the logging file and the weights here if you did not change the setting in config file

### folder : tools
Most of the file do the job as their name, except the file **utils.py** for storing checkpoints, **banet_train.py** reponsible for the training, **demo.py** responsible for the quick display of the result, and the evaluate.py reponsible for the evaluation. 

## Training
To train the model, you may use the instruction 
```
sh train.sh 
``` 
or the instruction 
```
python tools/banet_train.py --epoch-to-train 150 --name THE_NAME_TO_STORE_WEIGHT --finetune-from ./res/THE_NAME_OF_WEIGHT.pth
```

## Evaluation
```
python tools/evaluate.py --weight-path ./res/THE_NAME_OF_WEIGHT.pth
```
