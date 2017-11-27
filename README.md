
# ModelZoo 

This repo contains a collection of mostly third-party deep learning models that 
can be run with the [neon]™ frontend and [Intel® Nervana™ Graph] libraries. 

The specific categories include more detailed information on the various models. 
Below is a summary of the various tasks and the models referenced here that 
address one way of approaching those tasks.

To start learning about how to create your own models optimized for specific 
kinds of tasks and the mathematical operations needed to run those tasks, 
please browse the [Intel® MKL Cookbook].


### Image classification
   
#### [ImageNet]
  - Googlenet
  - VGG
  - Alexnet

#### CIFAR-10
  - [Deep Residual Network for CIFAR10]
  - [All CNN]

### Object localization
  - [Fast-RCNN]

### Scene Classification
  - [Deep Residual Network for Scene Classification]

### Image Captioning
  - [LSTM]

### [Deep reinforcement learning] 
#### Gaming
  - [Simple Deep Q-network]

### [NLP]
#### [bAbI] (Question Answering)
  - GRU/LSTM model

#### Sentiment classification
  - [classification using IMDB dataset on LSTM]

### [Video]
  - C3D (UCF101 dataset)

### Transfer Learning
  - Classification using multi-scale sampling (with [PASCALVOC] dataset)

### Speech
  - [Deep Speech 2] with Librispeech dataset


[neon]:https://github.com/NervanaSystems/neon
[Intel® Nervana™ Graph]:https://github.com/NervanaSystems/ngraph
[Intel® MKL Cookbook]:https://software.intel.com/en-us/mkl_cookbook
[ImageNet]:ImageClassification
[Deep Residual Network for CIFAR10]:ImageClassification/CIFAR10/DeepResNet
[All CNN]:CIFAR10/All_CNN
[Fast-RCNN]:ObjectLocalization/FastRCNN
[Deep Residual Network for Scene Classification]:SceneClassification/DeepResNet
[LSTM]:ImageCaptioning/LSTM
[Deep reinforcement learning]:DeepReinforcement
[Simple Deep Q-network]:DeepReinforcement/Gaming/simple_dqn
[NLP]:NLP
[bAbI]:NLP/QandA/bAbI
[classification using IMDB dataset on LSTM]:NLP/SentimentClassification/IMDB
[Video]:Video
[PASCALVOC]:TransferLearning/PASCALVOC
[Deep Speech 2]:Speech/DeepSpeech
