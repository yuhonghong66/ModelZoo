This is an implementation of this paper: M. Oquab et. al. Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks, CVPR 2014.

Here are some helpful hints for running the example:
1. Use this command to start a fresh new training run
./examples/transfer_learning.py -e10 -r13 -b gpu --save_path models/transfer_learning/v0.pkl --serialize 1 --history 20 > log/transfer_learning/train_v0.log 2>&1 &

2. Use this command to run test. Make sure that the number of epochs specified in this command with the -e option is zero. That ensures that neon will skip the training and jump directly to testing.
./examples/transfer_learning.py -e0 -r13 -b gpu --model_file models/transfer_learning/v0.pkl > log/transfer_learning/test_v0.log 2>&1 &

3. Training each epoch can take 4-6 hours if you are training on the full 5000 images of the training dataset. If you had to terminate your training job for some reason, you can always restart from the last saved epoch with this command.
./examples/transfer_learning.py -e10 -r13 -b gpu --save_path models/transfer_learning/v0.pkl --serialize 1 --history 20 --model_file models/transfer_learning/v0.pkl > log/transfer_learning/train_v0.log 2>&1 &

Please check our blogs for a discussion of this implementation.
