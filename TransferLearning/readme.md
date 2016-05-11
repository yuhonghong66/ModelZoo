This is an implementation of this paper:

"Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks",
M. Oquab et. al., CVPR 2014.
The paper can be downloaded from [ihttp://www.di.ens.fr/willow/pdfscurrent/oquab14cvpr.pdf](http://www.di.ens.fr/willow/pdfscurrent/oquab14cvpr.pdf).


A full description of this model can be found on the Nervana website ([link](http://www.nervanasys.com/blog/))


To start a fresh new training run that trains for 10 epochs:
```
python transfer_learning.py -e 10 -b gpu -s tlearn_save.prm --serialize 1
```
The script will download an alexnet model pretrained on the ILSVCR2012 dataset from the neon
Model Zoo to initialize the the feature extraction layers of this model.  Then training will
proceed on the PASCAL VOC dataset.

Training can take some time, make sure to serialize checkpoint using the "--serialize" option.
If training is interrupted, it can be resumed from the last saved checkpoint by adding the
"--model_file <last saved checkpoint file>" option to the training command above.

To calculate the classification errors on a saved trained model, run the same command
with the epochs set to 0 and the serialized model file as an input parameter. For
example:
```
python transfer_learning.py -e 0 -b gpu --model_file tlearn_save.prm
```

