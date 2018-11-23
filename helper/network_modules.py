import tensorflow as tf
import modules as m
import numpy as np



# activation functions
# -----


def softmax_cross_entropy(a, b, name):
  """custom loss function based on cross-entropy that can be used within the error module"""
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=b, name=name))

def sigmoid_cross_entropy(a, b, name):
  """custom loss function based on sigmoid cross-entropy that can be used within the error module"""
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=b, name=name))

def lrn_relu(x, name=None, depth_radius=5, bias=1, alpha=1e-4, beta=0.5):
  """custom activation function that combines ReLU with a Local Response Normalization and
  custom parameters"""
  return tf.nn.lrn(input=tf.nn.relu(x, name), depth_radius=5, bias=1, alpha=1e-4, beta=0.5, name=name)


# dynamic network constructor for varying depth
# -----

## not included in this release



# static 2 layer network
# -----
        
class ReCoNNet(m.ComposedModule):
  def define_inner_modules(self, name, is_training, activations, conv_filter_shapes, bias_shapes, ksizes, pool_strides, topdown_filter_shapes, topdown_output_shapes, keep_prob, FLAGS):

    # create all modules of the network
    # -----

    self.layers = {}
    with tf.name_scope('input_normalization'):
      self.layers["inp_norm"] = m.NormalizationModule("inp_norm")
    with tf.name_scope('convolutional_layer_0'):
      if FLAGS.batchnorm:
        self.layers["conv0"] = m.TimeConvolutionalLayerWithBatchNormalizationModule("conv0", bias_shapes[0][-1], is_training, 0.0, 1.0, 0.5, activations[0], conv_filter_shapes[0], [1,1,1,1], bias_shapes[0])
      else:
        self.layers["conv0"] = m.TimeConvolutionalLayerModule("conv0", activations[0], conv_filter_shapes[0], [1,1,1,1], bias_shapes[0])
    with tf.name_scope('lateral_layer_0'):
      lateral_filter_shape = conv_filter_shapes[0]
      tmp = lateral_filter_shape[2]
      lateral_filter_shape[2] = lateral_filter_shape[3]
      lateral_filter_shape[3] = tmp
      self.layers["lateral0"] = m.Conv2DModule("lateral0", lateral_filter_shape, [1,1,1,1])
      self.layers["lateral0_batchnorm"] = m.BatchNormalizationModule("lateral0_batchnorm", lateral_filter_shape[-1], is_training, beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5, moment_axes=[0,1,2], variance_epsilon=1e-3)
    with tf.name_scope('pooling_layer_0'):
      self.layers["pool0"] = m.MaxPoolingModule("pool0", ksizes[0], pool_strides[0])
    with tf.name_scope('dropout_layer_0'):
      self.layers['dropoutc0'] = m.DropoutModule('dropoutc0', keep_prob=keep_prob)
    with tf.name_scope('convolutional_layer_1'):
      if FLAGS.batchnorm:
        self.layers["conv1"] = m.TimeConvolutionalLayerWithBatchNormalizationModule("conv1", bias_shapes[1][-1], is_training, 0.0, 1.0, 0.5, activations[1], conv_filter_shapes[1], [1,1,1,1], bias_shapes[1])
      else:
        self.layers["conv1"] = m.TimeConvolutionalLayerModule("conv1", activations[1], conv_filter_shapes[1], [1,1,1,1], bias_shapes[1])
    with tf.name_scope('topdown_layer_0'):
      self.layers["topdown0"] = m.Conv2DTransposeModule("topdown0", topdown_filter_shapes[0], [1,2,2,1], topdown_output_shapes[0])
      self.layers["topdown0_batchnorm"] = m.BatchNormalizationModule("topdown0_batchnorm",topdown_output_shapes[0][-1], is_training, beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5, moment_axes=[0,1,2], variance_epsilon=1e-3)
    with tf.name_scope('lateral_layer_1'):
      lateral_filter_shape = conv_filter_shapes[1]
      tmp = lateral_filter_shape[2]
      lateral_filter_shape[2] = lateral_filter_shape[3]
      lateral_filter_shape[3] = tmp
      self.layers["lateral1"] = m.Conv2DModule("lateral1", lateral_filter_shape, [1,1,1,1])
      self.layers["lateral1_batchnorm"] = m.BatchNormalizationModule("lateral1_batchnorm", lateral_filter_shape[-1], is_training, beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5, moment_axes=[0,1,2], variance_epsilon=1e-3)
    with tf.name_scope('pooling_layer_1'):
      self.layers["pool1"] = m.MaxPoolingModule("pool1", ksizes[1], pool_strides[1])
      self.layers["flatpool1"] = m.FlattenModule("flatpool1")
    with tf.name_scope('dropout_layer_1'):
      self.layers['dropoutc1'] = m.DropoutModule('dropoutc1', keep_prob=keep_prob)
    with tf.name_scope('fully_connected_layer_0'):
       if FLAGS.batchnorm:
         self.layers["fc0"] = m.FullyConnectedLayerWithBatchNormalizationModule("fc0", bias_shapes[-1][-1], is_training, 0.0, 1.0, 0.5, activations[2], int(np.prod(np.array(bias_shapes[1]) / np.array(pool_strides[1]))), np.prod(bias_shapes[2]))
       else:
         self.layers["fc0"] = m.FullyConnectedLayerModule("fc0", activations[2], int(np.prod(np.array(bias_shapes[1]) / np.array(pool_strides[1]))), np.prod(bias_shapes[2]))


    # connect all modules of the network in a meaningful way
    # -----

    with tf.name_scope('wiring_of_modules'):
      self.layers["conv0"].add_input(self.layers["inp_norm"], 0)
      self.layers["pool0"].add_input(self.layers["conv0"])
      self.layers["dropoutc0"].add_input(self.layers["pool0"])
      self.layers["conv1"].add_input(self.layers["dropoutc0"], 0)
      self.layers["pool1"].add_input(self.layers["conv1"])
      self.layers["dropoutc1"].add_input(self.layers["pool1"])
      self.layers["flatpool1"].add_input(self.layers["dropoutc1"])
      self.layers["fc0"].add_input(self.layers["flatpool1"])
      if "L" in FLAGS.architecture:
        if FLAGS.batchnorm:
          self.layers["lateral0"].add_input(self.layers["conv0"].preactivation)
          self.layers["lateral0_batchnorm"].add_input(self.layers["lateral0"])
          self.layers["conv0"].add_input(self.layers["lateral0_batchnorm"], -1)
          self.layers["lateral1"].add_input(self.layers["conv1"].preactivation)
          self.layers["lateral1_batchnorm"].add_input(self.layers["lateral1"])
          self.layers["conv1"].add_input(self.layers["lateral1_batchnorm"], -1)
        else:
          self.layers["lateral0"].add_input(self.layers["conv0"].preactivation)
          self.layers["conv0"].add_input(self.layers["lateral0"], -1)
          self.layers["lateral1"].add_input(self.layers["conv1"].preactivation)
          self.layers["conv1"].add_input(self.layers["lateral1"], -1)
      if "T" in FLAGS.architecture:
        if FLAGS.batchnorm:
          self.layers["topdown0_batchnorm"].add_input(self.layers["topdown0"])
          self.layers["conv0"].add_input(self.layers["topdown0_batchnorm"], -1)
          self.layers["topdown0"].add_input(self.layers["conv1"].preactivation)
        else:
          self.layers["conv0"].add_input(self.layers["topdown0"], -1)
          self.layers["topdown0"].add_input(self.layers["conv1"].preactivation)
    with tf.name_scope('input_output'):
      self.input_module = self.layers["inp_norm"]
      self.output_module = self.layers["fc0"]
