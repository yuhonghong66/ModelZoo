
This is an implementation of a LSTM model to solve the IMDB sentiment classification task.


### Model script

The model script is included with the neon repo examples ([link](https://github.com/NervanaSystems/neon/blob/master/examples/imdb_lstm.py))

### Trained weights

The trained weights file can be downloaded from AWS using the following link:
[imdb_lstm.p][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/imdb_lstm/imdb_lstm.p

### neon version
The model weigth file above has been generated using neon version tag [v1.4.0]((https://github.com/NervanaSystems/neon/releases/tag/v1.4.0).
It may not work with other versions.

### Performance

This model is acheiving 94.7% and 84.8% accuracy on the training and validation set, respectively.

### Instructions


Commands should be run from the neon installation root directory.  If neon was installed into
a virtualenv, make sure that it is active first.

To train the model from scratch for 2 epochs, use the command:
```
python examples/imdb_lstm.py -b gpu -e 2 -eval 1 -r 0 -s imdb_lstm.p
```


## Citation

```
When Are Tree Structures Necessary for Deep Learning of Representations?
Jiwei Li, Dan Jurafsky and Eduard Hovy
http://arxiv.org/pdf/1503.00185v1.pdf
```


