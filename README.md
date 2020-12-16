# VQA
Visual question and answer

## Running the code
This model is based off https://github.com/Cyanogenoid/pytorch-vqa
Before running the code please download the relevant train, validation and test data from https://vizwiz.org/tasks-and-datasets/vqa/
Each section below tells you which scripts to run depending on whether the results for the Resnet-18 or
CNN are desired

## CNN 
We run the scripts in the following order to achieve the best results
```
image_preprocessing_CNN.py
```
This extracts features from the image
```
preprocess-QA.py
```
This extracts features from the questions asked
```
train_models_cnn.py
```
This trains the model for 50 epochs with L2 regularization 0.0005
```
view-logs.py logs/<log name>
```
This prints out the results

## ResNet-18

We run the scripts in the following order to achieve the best results
```
image_preprocessing_resnet18.py
```
This extracts features from the image
```
preprocess-QA.py
```
This extracts features from the questions asked
```
train_models.py
```
This trains the model for 50 epochs with L2 regularization 0.001
```
view-logs.py logs/<log name>
```
This prints out the results
