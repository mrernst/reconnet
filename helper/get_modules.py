import tensorflow as tf
import os
import gzip
import numpy
import urllib.request

SOURCE_URL = 'https://raw.githubusercontent.com/mrernst/modules-py/master/src/'
WORK_DIRECTORY = './helper/'

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


class MODULES:
  def __init__(self):
    modules_name = "modules.py"
    train_data_filename = maybe_download(modules_name)