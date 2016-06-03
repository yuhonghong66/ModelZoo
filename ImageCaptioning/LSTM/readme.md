##Information

name: LSTM image captioning model based on CVPR 2015 paper "[Show and tell: A neural image caption 
generator](http://arxiv.org/abs/1411.4555)" and code from Karpathy's 
[NeuralTalk](https://github.com/karpathy/neuraltalk).

model script: script is included with the examples in the neon repo
              [image_caption.py](https://github.com/NervanaSystems/neon/blob/master/examples/image_caption.py)

model weights: [image_caption_flickr8k.p][S3_WEIGHTS_FILE]
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/imagecaption/lstm/image_caption_flickr8k.p

neon commit: This model file has been tested with neon commit tag [v1.4.0](https://github.com/NervanaSystems/neon/tree/v1.4.0)


##Description
The LSTM model is trained on the [flickr8k dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html)
using precomputed VGG features from http://cs.stanford.edu/people/karpathy/deepimagesent/. Model details can be 
found in the following [CVPR-2015 paper](http://arxiv.org/abs/1411.4555):

    Show and tell: A neural image caption generator.
    O. Vinyals, A. Toshev, S. Bengio, and D. Erhan.  
    CVPR, 2015 (arXiv ref. cs1411.4555)

The model was trained for 15 epochs where 1 epoch is 1 pass over all 5 captions of each image. 
Training data was shuffled each epoch. To evaluate on the test set, download the model and weights, 
activate the neon virtualenv and from the root neon directory run:

     python examples/image_caption.py --model_file [path_to_weights]
        
To train the model from scratch for 15 epochs use the command:

     python examples/image_caption.py -e 15 -s image_caption_flickr8k.p


##Performance
For testing, the model is only given the image and must predict 
the next word until a stop token is predicted. Greedy search is 
currently used by just taking the max probable word each time. 
Using the bleu score evaluation script from 
https://raw.githubusercontent.com/karpathy/neuraltalk/master/eval/ and evaluating 
against 5 reference sentences the results are below.

| BLEU | Score |
| ---- | ----  |
| B-1  | 54.3  |
| B-2  | 35.7  |
| B-3  | 22.8  |
| B-4  | 14.7  |

A few things that were not implemented are beam search, l2 regularization, and ensembles. With these things, 
performance would be a bit better.
