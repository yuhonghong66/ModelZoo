
This is an implementation of a LSTM model to solve the IMDB sentiment classification task.


### Model script


The [imdb_lstm.py](https://github.com/nervanazoo/NervanaModelZoo/blob/master/NLP/SentimentClassification/IMDB/imdb_lstm.py)
script shows how to train the model.

### Trained weights

The trained weights file can be downloaded from AWS using the following link:
[imdb_lstm.p][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/imdb_lstm/imdb_lstm.p

### Performance

This model is acheiving 94.3% and 84.5% accuracy on the training and validation set, respectively.

### Instructions

Neon version: commit SHA
[e7ab2c2e2](https://github.com/NervanaSystems/neon/commit/e7ab2c2e27f113a4d36d17ba8c79546faed7d916).
Make sure that your local repo is synced to this commit and run the [installation procedure](http://neon.nervanasys.com/docs/latest/user_guide.html#installation) before proceeding.

If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.  Also, the commands below use the GPU backend by default so add `-b cpu` if you are running on a system without a compatible GPU.

To train the model from scratch for 2 epochs, use the command:
```
python imdb_lstm.py -b gpu -e 2 -eval 1 -r 0 -s imdb_lstm.p
```

Additional options are available to add features like saving checkpoints and displaying logging information, use the `--help` option for details.


## Citation

```
When Are Tree Structures Necessary for Deep Learning of Representations?
Jiwei Li, Dan Jurafsky and Eduard Hovy
http://arxiv.org/pdf/1503.00185v1.pdf
```


