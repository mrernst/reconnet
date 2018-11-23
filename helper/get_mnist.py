import tensorflow as tf
import os
import gzip
import numpy
import urllib.request

IMAGE_SIZE = 28
PIXEL_DEPTH = 255

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = './input/mnist/'

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = 2 * (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

def add_occluders(image_batch, batch_size, number_of_occluders, sig=2/28*5):
  #generate occluder gaussian
  x, y = numpy.meshgrid(numpy.linspace(-1,1,IMAGE_SIZE), numpy.linspace(-1,1,IMAGE_SIZE))
  x = numpy.repeat(x[...,numpy.newaxis], batch_size, axis=2)
  y = numpy.repeat(y[...,numpy.newaxis], batch_size, axis=2)
  
  for occ in range(number_of_occluders):
    sigma = numpy.ones(batch_size)*sig
    mux = numpy.random.uniform(-1,1,batch_size)
    muy = numpy.random.uniform(-1,1,batch_size) 
    d = numpy.sqrt((x-mux)*(x-mux)+(y-muy)*(y-muy))
    g = numpy.heaviside((d-sig), 0)
    # tests with gaussian
    #g = numpy.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    #g = numpy.heaviside(numpy.round(g,3), 0)
    #g = 1 - g
    g = numpy.expand_dims(numpy.swapaxes(g,0,-1), axis=-1)
    image_batch = numpy.multiply(image_batch,g)
  return image_batch

class MNIST:
  def __init__(self, img_height=IMAGE_SIZE, img_width=IMAGE_SIZE):
    train_images_name = "train-images-idx3-ubyte.gz"  #  training set images (9912422 bytes)
    train_data_filename = maybe_download(train_images_name)
    train_mnist = extract_data(train_data_filename, 60000)
    
    train_label_name = "train-labels-idx1-ubyte.gz"  #  training set labels (28881 bytes)
    train_label_filename = maybe_download(train_label_name)
    train_mnist_label = extract_labels(train_label_filename, 60000)
    
    test_image_name = "t10k-images-idx3-ubyte.gz"  #  test set images (1648877 bytes)
    test_data_filename = maybe_download(test_image_name)
    test_mnist = extract_data(test_data_filename, 5000)
    
    test_label_name = "t10k-labels-idx1-ubyte.gz"  #  test set labels (4542 bytes)
    test_label_filename = maybe_download(test_label_name)
    test_mnist_label = extract_labels(test_label_filename, 5000)
    
    self.train = Train(images=train_mnist, labels=train_mnist_label, binlabels=train_mnist_label, img_height=img_height, img_width=img_width)
    self.validation = Test(images=test_mnist, labels=test_mnist_label, binlabels=test_mnist_label, img_height=img_height, img_width=img_width)
    self.test = Test(images=test_mnist, labels=test_mnist_label, binlabels=test_mnist_label, img_height=img_height, img_width=img_width)
    print("Training size: {}, Validation Size: {}, Test Size: {}".format(len(self.train.images), len(self.validation.images), len(self.test.images)))


class Train:
  pt = 0
  def __init__(self, images, labels, binlabels, img_height, img_width):
      self.images = images
      self.onehot = list(map(lambda x: [1 if i == x else 0 for i in range(10)], labels[:]))
      self.nhot = list(map(lambda x: [1 if i == x else 0 for i in range(10)], labels[:])) #binlabels

  def next_batch(self, n):
      self.pt += n
      if self.pt > len(self.images):
          self.pt = 0
          raise EOFError("No more data in set")
          self.pt = n
      out = self.images[self.pt-n:self.pt], self.onehot[self.pt-n:self.pt], self.nhot[self.pt-n:self.pt]
      return out


class Test:
  pt = 0
  def __init__(self, images, labels, binlabels, img_height, img_width):
      self.images = images
      self.onehot = list(map(lambda x: [1 if i == x else 0 for i in range(10)], labels[:]))
      self.nhot = list(map(lambda x: [1 if i == x else 0 for i in range(10)], labels[:])) #binlabels

  def next_batch(self, n):
      self.pt += n
      if self.pt > len(self.images):
        self.pt = 0
        raise EOFError("No more data in set")
        self.pt = n
      out = self.images[self.pt-n:self.pt],  self.onehot[self.pt-n:self.pt], self.nhot[self.pt-n:self.pt]
      return out
