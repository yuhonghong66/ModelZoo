
This is an implementation of a Deep Speech 2 model based on the paper
[Amodei, et al., 2015](https://arxiv.org/pdf/1512.02595v1.pdf) 
on end-to-end speech recognition.


### Instructions

For complete instructions on preparing datasets and/or training the model, refer to the [README](https://github.com/NervanaSystems/deepspeech) in our deepspeech repository. 

In particular, to evaluate the model's performance, you need to follow the instructions on preparing a manifest file using Librispeech's [test-clean](http://www.openslr.org/12/) dataset. 

Once the manifest file is prepared, you can evaluate the model by running 

 ```
 python evaluate.py --manifest val:/path/to/test_data_manifest --model_file /path/to/saved_model
 ```

where /path/to/test_data_manifest should point to the location of the manifest file that you prepared for use with the test dataset and /path/to/saved_model should point to the location of the trained model.


### Trained weights

The trained model weights can be downloaded from AWS using the following link:
[trained speech model weights][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/Deep_Speech/Librispeech/librispeech_15_epochs.prm

The weights provded here have been verified to work with [neon version 1.7.0](https://github.com/NervanaSystems/neon/releases/tag/v1.7.0)
The weights file may not work with other versions of neon.

This set of weights was trained using our deepspeech repo's [train.py](https://github.com/NervanaSystems/deepspeech/blob/master/speech/train.py) script for 15 epochs. Simple argmax decoding using this set of weights gives a CER of 14% when evaluated using the [test-clean Librispeech dataset](http://www.openslr.org/12/). 
