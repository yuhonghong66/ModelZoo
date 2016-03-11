##Model

This is an implementation of a deep convolutional neural network model inspired by the paper 
[Springenberg, Dosovitskiy, Brox, Riedmiller 2014](http://arxiv.org/abs/1412.6806). 

### Model script
The model run script is included below [cifar10_allcnn.py](./cifar10_allcnn.py).

### Trained weights
The trained weights file can be downloaded from AWS 
[cifar10_allcnn_e350.p][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/cifar10_allcnn/cifar10_allcnn_e350.p

### Performance
This model is acheiving 89.5% top-1 accuracy on the validation data set.  This is done using zca whitened, 
global contrast normalized data, without crops or flips.  This is the same performance we achieve running the 
same model configuration and data through Caffe.  

### Instructions
This script was tested with the 
[neon commit SHA e7ab2c2e2](https://github.com/NervanaSystems/neon/commit/e7ab2c2e27f113a4d36d17ba8c79546faed7d916).  
Make sure that your local repo is synced to this commit and run the 
[installation procedure](http://neon.nervanasys.com/docs/latest/user_guide.html#installation) before proceeding.  


If neon is installed into a `virtualenv`, make sure that it is activated before running the commands below.  Also, the commands below use the GPU backend by default so add `-b cpu` if you are running on a system without a compatible GPU.


To test the model performance on the validation data set use the following command:
```
python cifar10_allcnn.py --model_file cifar10_allcnn_e350.p -eval 1
```

To train the model from scratch for 350 epochs, use the command:
```
python cifar10_allcnn.py -b gpu -e 350 -s cifar10_allcnn_trained.p
```

Additional options are available to add features like saving checkpoints and displaying logging information, 
use the `--help` option for details.


## Benchmarks

Machine and GPU specs:
```
Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz
Ubuntu 14.04.2 LTS
GPU: GeForce GTX TITAN X
CUDA Driver Version 7.0
```

The run times for the fprop and bprop pass are given in the table below.  The same model configuration is used in neon and caffe.  50 iterations are timed in each framework and only the mean value is reported. 


```
-------------------------------------------
|    Func     | neon (mean) | caffe (mean)|
-------------------------------------------
| fprop       |    14 ms    |    19 ms    |
| bprop       |    34 ms    |    65 ms    |
| update      |     3 ms    |    *        | 
| iteration   |    51 ms    |    85 ms    |
-------------------------------------------
* caffe update operation may be included in bprop or iteration time but is not individually timed.
```


## Citation

```
Jost Tobias Springenberg,  Alexey Dosovitskiy, Thomas Brox and Martin A. Riedmiller. 
Striving for Simplicity: The All Convolutional Net. 
arXiv preprint arXiv:1412.6806, 2014.
```
