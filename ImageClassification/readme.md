# Image classification

## ILSVRC2012 (ImageNet)

### Models

#### [Googlenet](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageClassification/ILSVRC2012/Googlenet)

This is an implementation of the GoogLeNet model for image classification described in
[Szegedy et. al. 2014](http://arxiv.org/pdf/1409.4842.pdf).

Citation:
```
Going deeper with convolutions
Szegedy, Christian; Liu, Wei; Jia, Yangqing; Sermanet, Pierre;
Reed, Scott; Anguelov, Dragomir; Erhan, Dumitru; Vanhoucke, Vincent;
Rabinovich, Andrew
arXiv:1409.4842
```

#### [VGG](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageClassification/ILSVRC2012/VGG)

We have adapted the 16 and 19 layer VGG model that is available for Caffe for use with neon.  

Citation:

```
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
```

#### [Alexnet](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageClassification/ILSVRC2012/Alexnet)

An implementation of an image classification model based on
[Krizhevsky, Sutskever and Hinton 2012](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).


Citation:
```
ImageNet Classification with Deep Convolutional Neural Networks
Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
Advances in Neural Information Processing Systems 25
eds.F. Pereira, C.J.C. Burges, L. Bottou and K.Q. Weinberger
pp. 1097-1105, 2012
```

### Data

To run these models the ILSVRC2012 data set will need to be downloaded and converted into the data format
required by the neon ImageLoader class.  For instructions on this please see the
[neon user guide](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet)


## CIFAR10

### Models

#### [Deep Residual Network](https://github.com/nervanazoo/NervanaModelZoo/tree/master/ImageClassification/CIFAR10/DeepResNet)

An implementation of deep residual networks as described in [He, Zhang, Ren, Sun 2015](http://arxiv.org/abs/1512.03385).

Citation:
```
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition
arXiv preprint arXiv:1512.03385, 2015.
```

#### All CNN Model

An implementation of a deep convolutional neural network model inspired by the paper
[Springenberg, Dosovitskiy, Brox, Riedmiller 2014](http://arxiv.org/abs/1412.6806). 

Citation:
```
Jost Tobias Springenberg,  Alexey Dosovitskiy, Thomas Brox and Martin A. Riedmiller. 
Striving for Simplicity: The All Convolutional Net. 
arXiv preprint arXiv:1412.6806, 2014.
```

### Data

The Deep Residual Network model
[readme.md](https://github.com/nervanazoo/NervanaModelZoo/blob/master/ImageClassification/CIFAR10/DeepResNet/README.md)
provides instructions on converting the CIFAR-10 dataset to the format required by the neon ImageLoader class. 
The All CNN model script will download the CIFAR-10 dataset automatically.

