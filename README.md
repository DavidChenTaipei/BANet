# BANet
This is my implement code of the paper **Bilateral attention decoder: A lightweight decoder for real-time semantic segmentation** and the sturcture of the code is based on the implementation of [BiSeNet](https://github.com/CoinCheung/BiSeNet) by CoinCheung.

The structure and all the configs are set as what author mentions in the paper.

Now, I'm going to do the brief introduction to each file
### file : best_miou.txt
The best miou during training will be record here and the weight of it will be stored with **NAME_best.pth**.

### file: lr_record.txt
The learning rate need to be set at here, and will be changed due to the learning rate policy.

### folder : Configs 
The file config.py is the file where the parameters are set.

### folder : lib
In the lib folder, the **base_dataset.py** and **cityscapes_cv2.py** is where the dataset to be prepared. After these, the file **logger.py** is to set the information during training and **lr_scheduler.py** and **meters.py** are for learning rate policy and to set the meters for traininf repectively.Moreover, there is a subfolder named **model** and the struture of the model is written here. 

### folder : res
The model will store the logging file and the weights here if you did not change the setting in config file

### folder : tools
Most of the file do the job as their name, except the file **utils.py** for storing checkpoints, **banet_train.py** reponsible for the training, **demo.py** responsible for the quick display of the result, and the evaluate.py reponsible for the evaluation. 

### Training
To train the model, you may use the instruction 
```
sh train.sh 
``` 
or the instruction 
```
python tools/banet_train.py --epoch-to-train 150 --name THE_NAME_TO_STORE_WEIGHT --finetune-from ./res/THE_NAME_OF_WEIGHT.pth
```

### Evaluation
```
python tools/evaluate.py --weight-path ./res/THE_NAME_OF_WEIGHT.pth
```
