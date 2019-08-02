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
      self.layers["pool1"] = m.MaxPoolingModule("pool1", ksizes[0], pool_strides[1])
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



class ReCoNNet_with_CAM(m.ComposedModule):
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
      self.layers["pool1"] = m.MaxPoolingModule("pool1", ksizes[0], pool_strides[1])
    with tf.name_scope('global_average_pooling'):
      self.layers['gap'] = m.GlobalAveragePoolingModule('gap')
    with tf.name_scope('fully_connected_layer_0'):
       if FLAGS.batchnorm:
         self.layers["fc0"] = m.FullyConnectedLayerWithBatchNormalizationModule("fc0", bias_shapes[-1][-1], is_training, 0.0, 1.0, 0.5, activations[2], bias_shapes[1][-1], np.prod(bias_shapes[2]))
       else:
         self.layers["fc0"] = m.FullyConnectedLayerModule("fc0", activations[2], bias_shapes[1][-1], np.prod(bias_shapes[2]))


    # connect all modules of the network in a meaningful way
    # -----

    with tf.name_scope('wiring_of_modules'):
      self.layers["conv0"].add_input(self.layers["inp_norm"], 0)
      self.layers["pool0"].add_input(self.layers["conv0"])
      self.layers["dropoutc0"].add_input(self.layers["pool0"])
      self.layers["conv1"].add_input(self.layers["dropoutc0"], 0)
      self.layers["pool1"].add_input(self.layers["conv1"])
      self.layers["gap"].add_input(self.layers["pool1"])
      self.layers["fc0"].add_input(self.layers["gap"])
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



class CAM_Net(m.ComposedModule):
  def define_inner_modules(self, name, is_training, activations, conv_filter_shapes, bias_shapes, ksizes, pool_strides, topdown_filter_shapes, topdown_output_shapes, keep_prob, FLAGS):

    # create all modules of the network
    # -----

    self.layers = {}
    with tf.name_scope('input_normalization'):
      self.layers["inp_norm"] = m.NormalizationModule("inp_norm")
    with tf.name_scope('convolutional_layer_0'):
      #self.layers["conv0"] = m.TimeConvolutionalLayerWithBatchNormalizationModule("conv0", bias_shapes[0][-1], is_training, 0.0, 1.0, 0.5, activations[0], [5,5,1,32], [1,1,1,1], [1,28,28,32])
      self.layers["conv0"] = m.TimeConvolutionalLayerModule("conv0", activations[0], [3,3,1,32], [1,1,1,1], [1,28,28,32])
    with tf.name_scope('pooling_layer_0'):
      self.layers["pool0"] = m.MaxPoolingModule("pool0", ksizes[0], pool_strides[0])
    with tf.name_scope('convolutional_layer_1'):
      #self.layers["conv1"] = m.TimeConvolutionalLayerWithBatchNormalizationModule("conv1", bias_shapes[1][-1], is_training, 0.0, 1.0, 0.5, activations[1], [5,5,32,64], [1,1,1,1], [1,28,28,64])
      self.layers["conv1"] = m.TimeConvolutionalLayerModule("conv1", activations[1], [5,5,32,64], [1,1,1,1], [1,14,14,64])
    with tf.name_scope('pooling_layer_1'):
      self.layers["pool1"] = m.MaxPoolingModule("pool1", ksizes[0], pool_strides[1])
    with tf.name_scope('global_average_pooling'):
      self.layers['gap'] = m.GapModule('gap')
    with tf.name_scope('fully_connected_layer_0'):
      self.layers["fc0"] = m.FullyConnectedModule("fc0", bias_shapes[1][-1], np.prod(bias_shapes[2]))


    # connect all modules of the network in a meaningful way
    # -----

    with tf.name_scope('wiring_of_modules'):
      self.layers["conv0"].add_input(self.layers["inp_norm"], 0)
      self.layers["pool0"].add_input(self.layers["conv0"])
      self.layers["conv1"].add_input(self.layers["pool0"], 0)
      self.layers["pool1"].add_input(self.layers["conv1"])
      self.layers["gap"].add_input(self.layers["pool1"])
      self.layers["fc0"].add_input(self.layers["gap"])
    with tf.name_scope('input_output'):
      self.input_module = self.layers["inp_norm"]
      self.output_module = self.layers["fc0"]



# static 2 layer network
# -----
        
class ReCoNNet_pwnorm(m.ComposedModule):
  def define_inner_modules(self, name, is_training, activations, conv_filter_shapes, bias_shapes, ksizes, pool_strides, topdown_filter_shapes, topdown_output_shapes, keep_prob, FLAGS):

    # create all modules of the network
    # -----

    self.layers = {}
    with tf.name_scope('input_normalization'):
      self.layers["pw_norm"] = m.PixelwiseNormalizationModule("pw_norm", [1]+topdown_output_shapes[0][1:])
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
      self.layers["pool1"] = m.MaxPoolingModule("pool1", ksizes[0], pool_strides[1])
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
      #self.layers["inp_norm"].add_input(self.layers["pw_norm"])
      self.layers["conv0"].add_input(self.layers["pw_norm"], 0)
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
      self.input_module = self.layers["pw_norm"]
      self.output_module = self.layers["fc0"]



# static 2 layer network
# -----
        
class ShadowNet(m.ComposedModule):
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
      self.layers["pool0"] = m.MaxPoolingWithArgmaxModule("pool0", ksizes[0], pool_strides[0])
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
      self.layers["pool1"] =  m.MaxPoolingWithArgmaxModule("pool1", ksizes[0], pool_strides[1])
      self.layers["flatpool1"] = m.FlattenModule("flatpool1")
    with tf.name_scope('dropout_layer_1'):
      self.layers['dropoutc1'] = m.DropoutModule('dropoutc1', keep_prob=keep_prob)
    with tf.name_scope('fully_connected_layer_0'):
       if FLAGS.batchnorm:
         self.layers["fc0"] = m.FullyConnectedLayerWithBatchNormalizationModule("fc0", bias_shapes[-1][-1], is_training, 0.0, 1.0, 0.5, activations[2], int(np.prod(np.array(bias_shapes[1]) / np.array(pool_strides[1]))), np.prod(bias_shapes[2]))
       else:
         self.layers["fc0"] = m.FullyConnectedLayerModule("fc0", activations[2], int(np.prod(np.array(bias_shapes[1]) / np.array(pool_strides[1]))), np.prod(bias_shapes[2]))
    
    with tf.name_scope('unpooling_layer_1'):
      self.layers["unpool1"] = m.UnpoolingModule("unpool1", ksizes[0], pool_strides[0])
    with tf.name_scope('unconvolution_layer_1'):
      self.layers["unconv1"] = m.UnConvolutionModule("unconv1", conv_filter_shapes[1],[1,1,1,1], topdown_output_shapes[0][:1]+ bias_shapes[1][1:])
    with tf.name_scope('unpooling_layer_0'):
      self.layers["unpool0"] = m.UnpoolingModule("unpool0", ksizes[0], pool_strides[0])
    with tf.name_scope('unconvolution_layer_0'):
      self.layers["unconv0"] = m.UnConvolutionModule("unconv0", conv_filter_shapes[0],[1,1,1,1],topdown_output_shapes[0])

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
      
      #try out unpooling
      self.layers["unpool1"].add_input(self.layers["pool1"])
      
      self.layers["unconv1"].add_input(self.layers["unpool1"])
      self.layers["unconv1"].add_input(self.layers["conv1"])
      
      self.layers["unpool0"].add_input(self.layers["unconv1"])
      self.layers["unpool0"].add_input(self.layers["pool0"])
      
      self.layers["unconv0"].add_input(self.layers["unpool0"])
      self.layers["unconv0"].add_input(self.layers["conv0"])
      
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
    
   
class AutoEncoderME1(m.ComposedModule):
  def define_inner_modules(self, name, is_training, trainable_input, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, keep_prob, FLAGS):
    # book-keeping of dimensions
    self.shapes = {}
    # create all modules of the network
    # -----
    
    self.layers = {}
    with tf.name_scope('input_normalization'):
      self.layers["inp_norm"] = m.NormalizationModule("inp_norm")
    with tf.name_scope('convolutional_layer_0'):
      self.layers["conv0"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv0", \
        32, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [3,3,IMAGE_CHANNELS,32], [1,2,2,1], [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 32])
      # self.layers["conv0"] = m.ConvolutionalLayerModule("conv0", \
      #   lrn_relu, [3,3,IMAGE_CHANNELS,32], [1,2,2,1], [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 32])
      
    # with tf.name_scope('pooling_layer_0'):
    #   self.layers["pool0"] = m.MaxPoolingModule("pool0", [1,2,2,1], [1,2,2,1])
    
    with tf.name_scope('convolutional_layer_1'):
      self.layers["conv1"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv1", \
        16, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [3, 3, 32, 16], [1,2,2,1], [1, IMAGE_HEIGHT//4, IMAGE_WIDTH//4, 16])
      # self.layers["conv1"] = m.ConvolutionalLayerModule("conv1", \
      #   lrn_relu, [3, 3, 32, 16], [1,2,2,1], [1, IMAGE_HEIGHT//4, IMAGE_WIDTH//4, 16])
      
    # with tf.name_scope('pooling_layer_1'):
    #   self.layers["pool1"] = m.MaxPoolingModule("pool1", [1,2,2,1], [1,2,2,1])
    
    with tf.name_scope('convolutional_layer_2'):
      self.layers["conv2"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv2", \
        4, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [3, 3, 16, 4], [1,2,2,1], [1, np.ceil(IMAGE_HEIGHT/8), np.ceil(IMAGE_WIDTH/8), 4])
      # self.layers["conv2"] = m.ConvolutionalLayerModule("conv2", \
      #   lrn_relu, [3, 3, 16, 8], [1,2,2,1], [1, np.ceil(IMAGE_HEIGHT/8), np.ceil(IMAGE_WIDTH/8), 8])
      
    # with tf.name_scope('pooling_layer_2'):
    #   self.layers["pool2"] = m.MaxPoolingModule("pool2", [1,2,2,1], [1,2,2,1])
    
    with tf.name_scope('trainable_input_canvas'):
      self.layers['bottleneck_switch'] = m.SwitchModule('bottleneck_switch', trainable_input)
      self.layers['bottleneck_canvas'] = m.BiasModule('bottleneck_canvas', [FLAGS.batchsize,np.ceil(IMAGE_HEIGHT/8), np.ceil(IMAGE_WIDTH/8), 4])
    
    with tf.name_scope('deconvolutional_layer_0'):
      self.layers["deconv0"] = m.Conv2DTransposeModule("deconv0", \
        [3,3,16,4], [1,2,2,1], [FLAGS.batchsize,IMAGE_HEIGHT//4,IMAGE_WIDTH//4,16])
      # self.layers["deconv0_bn"] = m.BatchNormalizationModule("deconv0_batchnorm", \
      #  8, is_training, beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5, moment_axes=[0,1,2], variance_epsilon=1e-3)
      self.layers["deconv0_act"] = m.ActivationModule("deconv0_act", \
        lrn_relu)
    with tf.name_scope('deconvolutional_layer_1'):
      self.layers["deconv1"] = m.Conv2DTransposeModule("deconv1", \
        [3,3,32,16], [1,2,2,1], [FLAGS.batchsize,IMAGE_HEIGHT//2,IMAGE_WIDTH//2,32])
      # self.layers["deconv0_bn"] = m.BatchNormalizationModule("deconv1_batchnorm", \
      # 16, is_training, beta_init=0.0, gamma_init=0.1, ema_decay_rate=0.5, moment_axes=[0,1,2], variance_epsilon=1e-3)
      self.layers["deconv1_act"] = m.ActivationModule("deconv1_act", \
        lrn_relu)
    with tf.name_scope('deconvolutional_layer_2'):
      self.layers["deconv2"] = m.Conv2DTransposeModule("deconv2", \
        [3,3,IMAGE_CHANNELS,32], [1,2,2,1], [FLAGS.batchsize,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
      
      self.layers["deconv2_act"] = m.ActivationModule("deconv2_act", \
        lrn_relu)
      
    # connect all modules of the network in a meaningful way
    # -----

    with tf.name_scope('wiring_of_modules'):
      self.layers["conv0"].add_input(self.layers["inp_norm"])
      self.layers["conv1"].add_input(self.layers["conv0"])
      self.layers["conv2"].add_input(self.layers["conv1"])
      self.layers["bottleneck_switch"].add_input(self.layers["conv2"])
      self.layers["bottleneck_switch"].add_input(self.layers["bottleneck_canvas"])
      
      
      self.layers["deconv0"].add_input(self.layers["bottleneck_switch"])
      self.layers["deconv0_act"].add_input(self.layers["deconv0"])
      
      self.layers["deconv1"].add_input(self.layers["deconv0_act"])
      self.layers["deconv1_act"].add_input(self.layers["deconv1"])
      
      self.layers["deconv2"].add_input(self.layers["deconv1_act"])
      self.layers["deconv2_act"].add_input(self.layers["deconv2"])
      

    with tf.name_scope('input_output'):
      self.input_module = self.layers["inp_norm"]
      self.output_module = self.layers["deconv2_act"]


class AutoEncoderCW1(m.ComposedModule):
  def define_inner_modules(self, name, is_training, trainable_input, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, keep_prob, FLAGS):
    # book-keeping of dimensions
    self.shapes = {}
    # create all modules of the network
    # -----
    
    self.layers = {}
    with tf.name_scope('input_normalization'):
      self.layers["inp_norm"] = m.NormalizationModule("inp_norm")
    with tf.name_scope('convolutional_layer_0'):
      self.layers["conv0"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv0", \
        16, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [4,4,IMAGE_CHANNELS,16], [1,2,2,1], [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 16])
    
    with tf.name_scope('convolutional_layer_1'):
      self.layers["conv1"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv1", \
        8, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 16, 8], [1,1,1,1], [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 8])
    
    with tf.name_scope('convolutional_layer_2'):
      self.layers["conv2"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv2", \
        4, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 8, 4], [1,1,1,1], [1, IMAGE_HEIGHT//2, IMAGE_HEIGHT//2, 4])
        
    with tf.name_scope('fc_layer0'):
      self.layers["flat0"] = m.FlattenModule("flat0")    
      self.layers["fc0"] = m.FullyConnectedLayerWithBatchNormalizationModule("fc0", 64, is_training, 0.0, 1.0, 0.5, tf.nn.relu, 784, 64)
    
    with tf.name_scope('bottleneck'):  
      self.layers['bottleneck_switch'] = m.SwitchModule('bottleneck_switch', trainable_input)
      self.layers['bottleneck_canvas'] = m.BiasModule('bottleneck_canvas', [FLAGS.batchsize, 64])
    
    with tf.name_scope('fc_layer1'):
      self.layers["fc1"] = m.FullyConnectedLayerWithBatchNormalizationModule("fc1", 784, is_training, 0.0, 1.0, 0.5, tf.nn.relu, 64, 784)
      self.layers["reshape0"] = m.ReshapeModule("reshape0", [FLAGS.batchsize, IMAGE_HEIGHT//2, IMAGE_HEIGHT//2, 4])    
      
    with tf.name_scope('convolutional_layer_3'):
      self.layers["conv3"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv3", \
        8, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 4, 8], [1,1,1,1], [1, IMAGE_HEIGHT//2, IMAGE_HEIGHT//2, 8])
    
    with tf.name_scope('convolutional_layer_4'):
      self.layers["conv4"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv4", \
        16, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 8, 16], [1,1,1,1], [1, IMAGE_HEIGHT//2, IMAGE_HEIGHT//2, 16])
    
    
    
    with tf.name_scope('deconvolutional_layer_0'):
      self.layers["deconv0"] = m.Conv2DTransposeModule("deconv0", \
        [4,4,IMAGE_CHANNELS,16], [1,2,2,1], [FLAGS.batchsize,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
      self.layers["deconv0_act"] = m.ActivationModule("deconv0_act", \
        lrn_relu)
      
    # connect all modules of the network in a meaningful way
    # -----

    with tf.name_scope('wiring_of_modules'):
      self.layers["conv0"].add_input(self.layers["inp_norm"])
      self.layers["conv1"].add_input(self.layers["conv0"])
      self.layers["conv2"].add_input(self.layers["conv1"])
      
      self.layers["flat0"].add_input(self.layers["conv2"])
      self.layers["fc0"].add_input(self.layers["flat0"])
      
      self.layers["bottleneck_switch"].add_input(self.layers["fc0"])
      self.layers["bottleneck_switch"].add_input(self.layers["bottleneck_canvas"])
      
      self.layers["fc1"].add_input(self.layers["bottleneck_switch"])
      self.layers["reshape0"].add_input(self.layers["fc1"])
      
      self.layers["conv3"].add_input(self.layers["reshape0"])
      self.layers["conv4"].add_input(self.layers["conv3"])
      
      self.layers["deconv0"].add_input(self.layers["conv4"])
      self.layers["deconv0_act"].add_input(self.layers["deconv0"])

      

    with tf.name_scope('input_output'):
      self.input_module = self.layers["inp_norm"]
      self.output_module = self.layers["deconv0_act"]

class AutoEncoderCW2(m.ComposedModule):
  def define_inner_modules(self, name, is_training, trainable_input, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, keep_prob, FLAGS):
    # book-keeping of dimensions
    self.shapes = {}
    # create all modules of the network
    # -----
    
    self.layers = {}
    with tf.name_scope('input_normalization'):
      self.layers["inp_norm"] = m.NormalizationModule("inp_norm")
    with tf.name_scope('convolutional_layer_0'):
      self.layers["conv0"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv0", \
        32, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [4,4,IMAGE_CHANNELS,32], [1,2,2,1], [1, IMAGE_HEIGHT//2, IMAGE_WIDTH//2, 32])
    
    with tf.name_scope('convolutional_layer_1'):
      self.layers["conv1"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv1", \
        16, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [2,2, 32, 16], [1,2,2,1], [1, IMAGE_HEIGHT//4, IMAGE_WIDTH//4, 16])
    
    with tf.name_scope('convolutional_layer_2'):
      self.layers["conv2"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv2", \
        4, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 16, 4], [1,1,1,1], [1, IMAGE_HEIGHT//4, IMAGE_HEIGHT//4, 4])
    
    with tf.name_scope('trainable_input_canvas'):
      self.layers['bottleneck_switch'] = m.SwitchModule('bottleneck_switch', trainable_input)
      self.layers['bottleneck_canvas'] = m.BiasModule('bottleneck_canvas', [FLAGS.batchsize,IMAGE_HEIGHT//4, IMAGE_HEIGHT//4, 4])
    
    
    with tf.name_scope('convolutional_layer_3'):
      self.layers["conv3"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv3", \
        16, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 4, 16], [1,1,1,1], [1, IMAGE_HEIGHT//4, IMAGE_HEIGHT//4, 16])
    
    with tf.name_scope('convolutional_layer_4'):
      self.layers["conv4"] = m.ConvolutionalLayerWithBatchNormalizationModule("conv4", \
        32, is_training, 0.0, 1.0, 0.5, \
        lrn_relu, [1, 1, 16, 32], [1,1,1,1], [1, IMAGE_HEIGHT//4, IMAGE_HEIGHT//4, 32])
    
    
    
    with tf.name_scope('deconvolutional_layer_0'):
      self.layers["deconv0"] = m.Conv2DTransposeModule("deconv0", \
        [4,4,IMAGE_CHANNELS,32], [1,4,4,1], [FLAGS.batchsize,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
      self.layers["deconv0_act"] = m.ActivationModule("deconv0_act", \
        lrn_relu)
      
    # connect all modules of the network in a meaningful way
    # -----

    with tf.name_scope('wiring_of_modules'):
      self.layers["conv0"].add_input(self.layers["inp_norm"])
      self.layers["conv1"].add_input(self.layers["conv0"])
      self.layers["conv2"].add_input(self.layers["conv1"])
      self.layers["bottleneck_switch"].add_input(self.layers["conv2"])
      self.layers["bottleneck_switch"].add_input(self.layers["bottleneck_canvas"])
      
      
      self.layers["conv3"].add_input(self.layers["bottleneck_switch"])
      self.layers["conv4"].add_input(self.layers["conv3"])
      
      self.layers["deconv0"].add_input(self.layers["conv4"])
      self.layers["deconv0_act"].add_input(self.layers["deconv0"])

      

    with tf.name_scope('input_output'):
      self.input_module = self.layers["inp_norm"]
      self.output_module = self.layers["deconv0_act"]
