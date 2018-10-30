### README for ReCoNNet
ReCoNNet is short for Recurrent Convolutional Neural Network

#### Information

This is an implementation in tensorflow of the network models used in the thesis "Recurrent Convolutional Neural Network for Occluded Object Recognition". In order to use it you need to have the following packages installed.

* tensorflow (>1.4)
*
*


#### Usage

To use the network simply clone the repository and start with
python3 reconnet.py

The MNIST dataset should download automatically and classification starts. In order to tweak the parameters like learningrate, batch size etc. take a look at the arguments to reconnet.py, e.g.

python3 reconnet.py --batchsize 25 --architecture BLT

uses the BLT architecture and a batch size of 25 instead of the default of 100.
