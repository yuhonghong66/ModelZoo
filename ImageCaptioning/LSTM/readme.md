##Information

name: LSTM image captioning model based on CVPR 2015 paper "[Show and tell: A neural image caption 
generator](http://arxiv.org/abs/1411.4555)" and code from Karpathy's 
[NeuralTalk](https://github.com/karpathy/neuraltalk).

model_script: [image_caption.py](https://github.com/nervanazoo/NervanaModelZoo/blob/master/ImageCaptioning/LSTM/image_caption.py)

model_weights: [image_caption_flickr8k.p][S3_WEIGHTS_FILE]

[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/imagecaption/lstm/image_caption_flickr8k.p

neon_commit: [e7ab2c2e2](https://github.com/NervanaSystems/neon/commit/e7ab2c2e27f113a4d36d17ba8c79546faed7d916)


##Description
The LSTM model is trained on the [flickr8k dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html)
using precomputed VGG features from http://cs.stanford.edu/people/karpathy/deepimagesent/. Model details can be 
found in the following [CVPR-2015 paper](http://arxiv.org/abs/1411.4555):

    Show and tell: A neural image caption generator.
    O. Vinyals, A. Toshev, S. Bengio, and D. Erhan.  
    CVPR, 2015 (arXiv ref. cs1411.4555)

The model was trained for 15 epochs where 1 epoch is 1 pass over all 5 captions of each image. 
Training data was shuffled each epoch. To evaluate on the test set, download the model and weights, 
and run:

        python image_caption.py --model_file [path_to_weights]
        
To train the model from scratch for 15 epochs use the command:

     python image_caption.py -i 1 -e 15 -s image_caption_flickr8k.p


##Performance
For testing, the model is only given the image and must predict 
the next word until a stop token is predicted. Greedy search is 
currently used by just taking the max probable word each time. 
Using the bleu score evaluation script from 
https://raw.githubusercontent.com/karpathy/neuraltalk/master/eval/ and evaluating 
against 5 reference sentences the results are below.

| BLEU | Score |
| ---- | ----  |
| B-1  | 52.0  |
| B-2  | 34.0  |
| B-3  | 21.5  |
| B-4  | 13.9  |

A few things that were not implemented are beam search, l2 regularization, and ensembles. With these things, 
performance would be a bit better.
