import numpy as np
import os, sys
from scipy.io import savemat, loadmat

sys.path.insert(0, './helper/')
from get_mnist import MNIST
#import tensorflow as tf


def pad_img(img, shape=[1000, 40, 40, 1]):
    result = np.zeros(shape)
    result[:, 4:32, 4:32, 0] = img
    return result


def random_crop(img, shape=36):
    i, j = np.random.choice(range(5), 2)
    return img[i: i + shape, j: j + shape, :]


def _crop_merge(xi, xo):
    xi = random_crop(xi)
    xo = random_crop(xo)
    combined_img = np.concatenate([xi, xo], -1)
    combined_img = np.max(combined_img, -1, keepdims=True)
    return combined_img
    

class MultiMNISTBuilder(object):
    def __init__(self, num_class=10, num_pro_digit=1):
        #(x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()
        mnist_data = MNIST()
        x,y = mnist_data.train.images[:,:,:,0], mnist_data.train.labels
        x_t, y_t = mnist_data.test.images[:,:,:,0], mnist_data.train.labels
        self.num_class = num_class
        self.num_pro_digit = num_pro_digit
        self.H = 4
        self.W = 4

        self._build(x, y, 'input/multimnist/MultiMNIST_index_train.mat')
        self._build(x_t, y_t, 'input/multimnist/MultiMNIST_index_test.mat')

    def _build(self, x, y, oFilename):
        N = x.shape[0]
        P = 3
        output = np.zeros([N, self.num_pro_digit, P], dtype=np.int64)
        index_all = set(range(N))
        index_of_class = [np.where(y==i)[0] for i in range(self.num_class)]
        counter = 1
        picture = np.zeros([N*self.num_pro_digit,36,36,1])
        label = np.zeros([N*self.num_pro_digit,2])
        
        for k in range(self.num_class):
            index_of_other_class = list(index_all - set(index_of_class[k]))
            for i in index_of_class[k]:
                print('\rProcessing {:5d}/{:5d}'.format(counter, N), end='')
                output[i, :, 0] = np.random.choice(
                    index_of_other_class, self.num_pro_digit, replace=False)                
                #produce pictures
                x[i,:,:] #basic image
                base = np.tile(x[i,:,:], (self.num_pro_digit,1,1)) # 1000x the same images tiled
                overlay = x[output[i,:,0],:,:] #number of combinative images (1000)
                padded_base = pad_img(base, shape=[self.num_pro_digit, 40,40,1])
                padded_overlay = pad_img(overlay, shape=[self.num_pro_digit, 40,40,1])
                for v in range(self.num_pro_digit):
                  picture[(counter-1)*self.num_pro_digit+v] = _crop_merge(padded_base[v], padded_overlay[v])
                  label[(counter-1)*self.num_pro_digit+v] = k, y[output[i,v,0]]
                counter += 1
                
                
                      
        # write to matlab file
        dictionary = {'images':picture, 'labels':label}
        if not os.path.exists(oFilename):
            os.makedirs(oFilename)
        savemat(oFilename, dictionary, do_compression=True)





if __name__ == '__main__':
    builder = MultiMNISTBuilder()