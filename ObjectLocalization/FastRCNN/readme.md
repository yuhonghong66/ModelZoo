
This is an implementation of a Fast-RCNN model inspired by the paper
[Ross Girshick 2015](http://arxiv.org/pdf/1504.08083v2.pdf) and [code](https://github.com/rbgirshick/fast-rcnn)
used for object detection.

The model scripts are part of the neon examples and can be accessed from this link to the
[neon repository](https://github.com/NervanaSystems/neon).  Specifically, the Fast RCNN example
readme and scripts can be found [here](https://github.com/NervanaSystems/neon/tree/master/examples/fast-rcnn).


### Instructions

Follow the neon [installation procedure](http://neon.nervanasys.com/docs/latest/installation.html)
before proceeding.

If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.
Also, the commands below use the GPU backend by default so add `-b cpu` if you are running on a system without
a compatible GPU.

The Fast RCNN scripts and [README](https://github.com/NervanaSystems/neon/blob/master/examples/fast-rcnn/README.md)
are located in the neon examples [directory](https://github.com/NervanaSystems/neon/tree/master/examples/fast-rcnn).

To measure the performance of the trained file (see above) using the Pascal VOC test data set, activate the 
neon virtual env (if applicable) and run the following command from the neon repo root directory:
```
python examples/fast-rcnn/test.py --model_file frcn_vgg.p
```


### Trained weights

The trained model weights can be downloaded from AWS using the following link:
[trained Fast-RCNN model weights][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/Fast_RCNN/frcn_vgg.p

The weights provded here have been verified to work with [neon version 1.4.0](https://github.com/NervanaSystems/neon/releases/tag/v1.4.0)
The weights file may not work with other versions of neon.

This set of weights was trained using the [train.py](/home/jenkins/jenkins_backup_6_2_2016.tar.gz) script for 20 epochs and
reaches a mAP of 0.56 on the vaidation data set.
