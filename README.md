# VARACNet
VARACNet is an ensemble of multiple Convolutional Neural Networks (CNN) finely tuned for detection and localization of traffic light. 

## DATASET
[Nexar Challenge Dataset](https://challenge.getnexar.com/challenge-1)

We also perform localization experiments using Faster R-CNN on a separate annotated dataset (UCSD traffic lights dataset) to demonstrate high performance for classification and detection tasks.

## Network Architecture
[Ensemble model architecture](https://raw.githubusercontent.com/Adityav2410/VARACNet/master/assets/images/model.png)

Each CNN in the model is built with the motive of giving superior performance while keeping the model size small. The sub-models have no more than 490k parameters but each achieves an accuracy greater than 87%. Models are tested and trained on the Nexar traffic lights challenge dataset with the aim of correctly recognizing the presence and state of traffic lights in images taken by the drivers using the Nexar app. We show that minimizing the number of parameters in each of the models allows quick training even when computational resources are not abundant.


## RESULTS

<img src="https://github.com/Adityav2410/VARACNet/blob/master/assets/images/dayTraffic1.png" width=350 align="middle" >          <img src="https://github.com/Adityav2410/VARACNet/blob/master/assets/images/dayTraffic2.png" width=350 align="middle" >



<img src="https://github.com/Adityav2410/VARACNet/blob/master/assets/images/nightTraffic.png" width=350 align="middle" >

### Evaluation Metrics

|    Model Name | Classification Accuracy(%) | Number of Parameters |  Challenge score | 
|:-------------:|:--------------------------:|:--------------------:|:----------------:|
|   Model 1     |            88.54           |        261,923       |       0.8838     |
|   Model 2     |            89.90           |        483,135       |       0.894      |
|   Model 3     |            89.35           |        442,403       |       0.8907     |
|   Model 4     |            88.1            |        640,163       |       0.8771     |
|   SqueezeNet  |            87.7            |        712,697       |       0.8726     |
|   VARACNet    |            91.7            |        1,827,624     |       0.9053     |
