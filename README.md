### README for ReCoNNet
ReCoNNet is short for Recurrent Convolutional Neural Network

#### Information

This is a tensorflow implementation of the network models used in the thesis "Recurrent Convolutional Neural Network for Occluded Object Recognition". In order to use it you need to have the following packages installed.

* tensorflow (>1.4)


#### Usage

To use the network simply clone the repository and start with
python3 reconnet.py

The modules-py framework and the MNIST dataset downloads automatically and classification starts. In order to tweak the parameters like learningrate, batch size etc. take a look at the arguments to reconnet.py.

python3 reconnet.py --batchsize 25 --architecture BLT

for example uses the BLT architecture and a batch size of 25 instead of the default of 100.
