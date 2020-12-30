# Global Wheat Detection

This is a task of object detection with [Global Wheat Head dataset](https://www.kaggle.com/c/global-wheat-detection). It contains a set of outdoor images of wheat plants around the world. There are 3,422 training images and  about 1,000 test images. In training time, we divide the training set into 5 fold, 4 for training and 1 for validation. Use [YOLOv5](https://github.com/ultralytics/yolov5) model from GitHub and train with pretrained checkpoint. The highest testing mAP can reach 75.38% in public LB and 65.75% in private LB.

# Reproducing Submission
1. [Installation](#Installation)
2. [Dataset Configuration](#Dataset-Configuration)
3. [Training](#Training) 
4. [Submission](#Submission)

To reproduct my submission without retrainig, you can see [Submission](#Submission) section directly.

# Installation
1. Clone this repository. 
```
git clone https://github.com/Yunyung/Global-Wheat-Detection-Competition
```

2. We will use [YOLOv5](https://github.com/ultralytics/yolov5) model from GitHub. cd to this repository and clone YOLOv5 repository.
```
cd Global-Wheat-Detection-Competition
git clone https://github.com/ultralytics/yolov5 # repository should be yolov5 v3.0

# or specify to clone yolov5 v3.0 
git clone -b v3.0 https://github.com/ultralytics/yolov5
```

3. Install the packages required by yolov5 with requirements.txt
```
cd yolov5
pip install -r requirements.txt
```




# Dataset Configuration
To train [Global Wheat Head dataset](https://www.kaggle.com/c/global-wheat-detection) Dataset on YOLOv5, we need to set up the configuration for it.
1. Download dataset from [Kaggle](https://www.kaggle.com/c/global-wheat-detection/data).
2. Unzip the data files and set the data directory structure as:
```
Global-Wheat-Detection-Competition
|__global-wheat-detection # dataset
|   |__train
|   |   |__<train_image>
|   |   |__ ...
|   |   |__ ... 
|   |
|   |__test
|   |   |__<test_image>
|   |   |__ ...
|   |   |__ ... 
|   |
|   |__train.csv
|   |__sample_submission.csv
|   
|__yolov5 
|   |__ ...
|   
|__config # config for training
|   |__ wheat0.yaml
|   |__ wheat0_trainOnKaggle.yaml
|   |__ yolov5x.yaml
|   |__ hyp.e1.yaml
|   
|__parser_data.py # data clean and split data to train/val set
|__yolov5-wheat-detection-2-stage-PL.ipynb # Kernal submmit to Kaggle
```
3. Parse data into train/val set.
```
python parse_data.py
``` 

# Training

### Train
Training command:
```
python train.py --img 1024 --batch 4 --epochs 100 --data ../config/wheat0.yaml --cfg ../config/yolov5x.yaml --weights yolov5x.pt --name yolo5x
```
If your GPUs are out of memory, please decrease batch size or change to smaller model like yolov51, yolov5m or yolov5s. The way of changing configuration setting is simlar to yolov5x, please check [Model Configuration](#Model-Configuration) section above.

###  Hyperparamters
#### Data Pre-process and Augmentation
*	Random mosaic
    * Resize image (shape=(1024, 1024))
    * Random translation (translate=0.1)
    * Random scale (scale=0.5)
    * Random horizontal flip
    * Random vertiacl flip

You can edit ```yolov5/data/hyp.scratch.yaml``` for changing augmentation parameters or using the other augmentation methods.

#### Other 
*	Epochs = 100
*	Batch size = 4
*	Optimizer = SGD (learning rate=0.01, momentum=0.937, weight decay=0.0005)
*	Warmup (epochs=3.0, momentum=0.8, bias learning rate=0.1)
*	Scheduler = cosine learning rate decay (final OneCycleLR learning rate=0.2)
*	Loss function = CIOU loss
*	Box loss gain = 0.05
*	Class loss gain = 0.5
*	Object loss gain = 1.0


You can edit ```yolov5/data/hyp.scratch.yaml``` for changing some hyperparameters and some need to specify at training time.


# Submission
To repoduce the result on Kaggle without training , please follow the steps below:
1. New a notebook on Kaggle.
2. Choose File/upload to upload the submission notebook ```yolov5-wheat-detection-2-stage-PL.ipynb``` .
3. Choose add data on the right hand side and add the datasets below:
   * WBF: [weightedboxesfusion](https://www.kaggle.com/shonenkov/weightedboxesfusion)
   * Configuration files: [configYolo5](https://www.kaggle.com/yunyung/configyolo5)
   * Trained weights: [yolov5-wheat](https://www.kaggle.com/yunyung/yolov5-wheat)
   * Yolov5 repository (with an additional file ```train_on_kaggle.py```): [yolov5](https://www.kaggle.com/yunyung/yolov5)
5. Save the notebook.
6. Submit the notebook.

If you have trained your own weights, you need to upload your weights, and change the path of the .pt file.
# Reference
*	ultralytics, yolov5, viewed 25 Nov 2020, [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
*    nvnn, YOLOv5 Pseudo Labeling, viewed 28 Dec 2020 , https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling?fbclid=IwAR0OlK0DmXEy2wCLR0MWiuUZ0exbrCOt7b4b6WdJwaQqyqNhLaBT63y7yPk

