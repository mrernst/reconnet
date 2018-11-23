# implementation of a Recurrent Neural Network inspired by Spoerer, McClure and Kriegeskorte
# Markus Ernst, October 2018


# import standard libraries
# -----

from __future__ import division
import sys, os, re
import tensorflow as tf
import numpy as np


# import custom written functions and classes
# -----
sys.path.insert(0, './helper/')
import get_mnist
import get_modules
get_modules.MODULES()

import modules as m
import network_modules as networks

# commandline arguments
# -----

# common flags
tf.app.flags.DEFINE_string('output_dir', './output/',
                           'output directory')
tf.app.flags.DEFINE_string('exp_name', 'noname_experiment',
                           'name of experiment')

tf.app.flags.DEFINE_string('name', '',
                           'name of the run')
tf.app.flags.DEFINE_string('dataset', 'mnist',
                           'use different datasets (...)')
tf.app.flags.DEFINE_boolean('restore_ckpt', True,
                           'indicate whether you want to restore from checkpoint')
tf.app.flags.DEFINE_integer('batchsize', 100,
                            'batchsize to use, default 100')
tf.app.flags.DEFINE_integer('epochs', 100,
                           'learn for n epochs')
tf.app.flags.DEFINE_integer('testevery', 1,
                           'test every n epochs')
tf.app.flags.DEFINE_integer('timedepth_beyond', 0,
                           'timedepth unrolled beyond trained timedepth, makes it possible evaluate the network on timesteps further than the network was trained')
tf.app.flags.DEFINE_boolean('verbose', True,
                           'generate verbose output')
tf.app.flags.DEFINE_integer('writeevery', 1,
                           'write every n timesteps to tfevent')

# database specific
tf.app.flags.DEFINE_string('input_type', 'monocular',
                           '(monocular, binocular) indicate whether you want to use a stereo dataset')
tf.app.flags.DEFINE_string('label_type', 'onehot',
                           '(onehot, nhot) indicate whether you want to use onehot oder nhot encoding')
tf.app.flags.DEFINE_integer('n_occluders', 5,
                           'number of occluding objects')
# architecture specific
tf.app.flags.DEFINE_string('architecture', 'BL',
                           'architecture of the network, see Spoerer paper: B, BF, BK, BL, BT, BLT')
tf.app.flags.DEFINE_integer('network_depth', 2,
                           'depth of the deep network')
tf.app.flags.DEFINE_integer('timedepth', 3,
                           'timedepth of the recurrent architecture')
tf.app.flags.DEFINE_integer('feature_mult', 1,
                            'feature multiplier, layer 2 does have feature_mult * features[layer1]')
tf.app.flags.DEFINE_boolean('batchnorm', True,
                           'indicate whether BatchNormalization should be used, default True')
tf.app.flags.DEFINE_float('learning_rate', 0.003,
                           'specify the learning rate')
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                           'dropout parameter for regularization, between 0 and 1.0, default 1 (off)')

FLAGS = tf.app.flags.FLAGS


# define correct network parameters
# -----

BATCH_SIZE = FLAGS.batchsize
TIME_DEPTH = FLAGS.timedepth
TIME_DEPTH_BEYOND = FLAGS.timedepth_beyond
N_TRAIN_EPOCH = FLAGS.epochs

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

CLASSES = 10


INP_MIN = -1
INP_MAX =  1
DTYPE = tf.float32

TEST_SUMMARIES = []
TRAIN_SUMMARIES = []
ADDITIONAL_SUMMARIES = []
IMAGE_SUMMARIES = []


inp = m.PlaceholderModule("input", (BATCH_SIZE, IMAGE_HEIGHT, \
  IMAGE_WIDTH, IMAGE_CHANNELS), dtype='uint8')
labels = m.PlaceholderModule("input_labels", (BATCH_SIZE, \
  CLASSES), dtype=DTYPE)
keep_prob = m.ConstantPlaceholderModule("keep_prob", shape=(), dtype=DTYPE)
is_training = m.ConstantPlaceholderModule("is_training", shape=(), dtype=tf.bool)


# global_step parameter for restarting and to continue training
global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.assign_add(global_step,1, name='increment_global_step')
global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
increment_global_epoch = tf.assign_add(global_epoch,1, name='increment_global_epoch')

lrate = tf.Variable(FLAGS.learning_rate, trainable=False, name='learning_rate')


# index for getting nhot or onehot labels
label_index = 1
CROSSENTROPY_FN = networks.softmax_cross_entropy
if FLAGS.label_type == 'nhot':
  label_index += 1
  CROSSENTROPY_FN = networks.sigmoid_cross_entropy

RECEPTIVE_PIXELS = 3
N_FEATURES = 32
FEATURE_MULTIPLIER = FLAGS.feature_mult

if "F" in FLAGS.architecture:
    N_FEATURES = 64
if "K" in FLAGS.architecture:
    RECEPTIVE_PIXELS = 5

activations = [networks.lrn_relu, networks.lrn_relu, tf.identity]
conv_filter_shapes = [[RECEPTIVE_PIXELS,RECEPTIVE_PIXELS,IMAGE_CHANNELS,N_FEATURES],[RECEPTIVE_PIXELS,RECEPTIVE_PIXELS,N_FEATURES,FEATURE_MULTIPLIER*N_FEATURES]]
bias_shapes = [[1,IMAGE_HEIGHT,IMAGE_WIDTH,N_FEATURES],[1,int(np.ceil(IMAGE_HEIGHT/2)),int(np.ceil(IMAGE_WIDTH/2)),FEATURE_MULTIPLIER*N_FEATURES], [1,CLASSES]]
ksizes = [[1,2,2,1],[1,IMAGE_HEIGHT//2,IMAGE_WIDTH//2,1]]
pool_strides = [[1,2,2,1],[1,2,2,1]]
topdown_filter_shapes = [[2,2,IMAGE_CHANNELS,FEATURE_MULTIPLIER*N_FEATURES]]
topdown_output_shapes = [[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS]]


# handle input/output directies
# -----

# check directories
DATASET_DIRECTORY = './input/{}/'.format(FLAGS.dataset)
EVAL_DIRECTORY = './'
RESULT_DIRECTORY = FLAGS.output_dir + '{}/{}/{}/{}/{}/'.format(FLAGS.exp_name, FLAGS.name, FLAGS.dataset, FLAGS.label_type, FLAGS.input_type)


# new result directory
RESULT_DIRECTORY = '{}/{}/{}/'.format(FLAGS.output_dir, FLAGS.exp_name, FLAGS.name)

# architecture string
ARCHITECTURE_STRING = ''
ARCHITECTURE_STRING += '{}{}_{}layer_fm{}_d{}'.format(FLAGS.architecture, FLAGS.timedepth, FLAGS.network_depth, FLAGS.feature_mult, 1.0)
if FLAGS.batchnorm:
  ARCHITECTURE_STRING += '_bn1'
else:
  ARCHITECTURE_STRING += '_bn0'
ARCHITECTURE_STRING += '_bs{}'.format(FLAGS.batchsize)
ARCHITECTURE_STRING += '_lr{}'.format(FLAGS.learning_rate)
# data string
DATA_STRING = ''
DATA_STRING += "{}_0occ_0p_0cm".format(FLAGS.dataset)
# format string
FORMAT_STRING = ''
FORMAT_STRING += '{}x{}'.format(IMAGE_HEIGHT, IMAGE_WIDTH)
FORMAT_STRING += "_{}_grayscale_{}".format(FLAGS.input_type, FLAGS.label_type)

RESULT_DIRECTORY += "{}/{}/{}/".format(ARCHITECTURE_STRING,DATA_STRING,FORMAT_STRING)
CHECKPOINT_DIRECTORY = RESULT_DIRECTORY + 'checkpoints/'

# make sure the directories exist, otherwise create them
if not os.path.exists(CHECKPOINT_DIRECTORY):
    os.makedirs(CHECKPOINT_DIRECTORY)


# get image data
# -----

# parse data from files using get_digits
dataset = get_mnist.MNIST()


# initialize classes with parameters
# -----

network = networks.ReCoNNet("ReCoNNet", is_training.placeholder, activations, conv_filter_shapes, bias_shapes, ksizes, pool_strides, topdown_filter_shapes, topdown_output_shapes, keep_prob.placeholder, FLAGS)

one_time_error = m.ErrorModule("cross_entropy", CROSSENTROPY_FN)
error = m.TimeAddModule("add_error")
optimizer = m.OptimizerModule("adam", tf.train.AdamOptimizer(lrate))
accuracy = m.BatchAccuracyModule("accuracy")


network.add_input(inp)
one_time_error.add_input(network)
one_time_error.add_input(labels)
error.add_input(one_time_error, 0)
error.add_input(error, -1)
optimizer.add_input(error)
accuracy.add_input(network)
accuracy.add_input(labels)


error.create_output(TIME_DEPTH + TIME_DEPTH_BEYOND)
optimizer.create_output(TIME_DEPTH)

for time in range(0,(TIME_DEPTH + TIME_DEPTH_BEYOND + 1)):
  accuracy.create_output(time)

if FLAGS.label_type == 'nhot':
  accuracy = m.NHotBatchAccuracyModule("accuracy", all_labels_true=True)
  accuracy_labels = m.NHotBatchAccuracyModule("accuracy_labels", all_labels_true=False)

  accuracy.add_input(network)
  accuracy.add_input(labels)
  accuracy_labels.add_input(network)
  accuracy_labels.add_input(labels)

  for time in range(0,(TIME_DEPTH + TIME_DEPTH_BEYOND + 1)):
    accuracy.create_output(time)
    accuracy_labels.create_output(time)



# average accuracy and error at mean test-time (maybe solve more intelligently) + embedding stuff
# -----

total_test_accuracy = {}
total_test_loss = {}

for time in accuracy.outputs:
  total_test_accuracy[time] = tf.Variable(0.)
  total_test_loss[time] = tf.Variable(0.)
count = tf.Variable(0.)

update_total_test_accuracy = {}
update_total_test_loss = {}

for time in accuracy.outputs:
  update_total_test_accuracy[time] = tf.assign_add(total_test_accuracy[time], accuracy.outputs[time])
  update_total_test_loss[time] = tf.assign_add(total_test_loss[time], error.outputs[time])
update_count = tf.assign_add(count, 1.)

reset_total_test_accuracy = {}
reset_total_test_loss = {}

for time in accuracy.outputs:
  reset_total_test_loss[time] = tf.assign(total_test_loss[time], 0.)
  reset_total_test_accuracy[time] = tf.assign(total_test_accuracy[time], 0.)
reset_count = tf.assign(count, 0.)

average_accuracy = {}
average_cross_entropy ={}
for time in accuracy.outputs:
  average_cross_entropy[time] = total_test_loss[time] / count
  average_accuracy[time] = total_test_accuracy[time] / count


# needed because tf 1.4 does not support grouping dict of tensor
update_accloss = tf.stack((list(update_total_test_loss.values()) + list(update_total_test_accuracy.values())))
reset_accloss = tf.stack((list(reset_total_test_accuracy.values()) + list(reset_total_test_loss.values())))

#confusion matrix
total_confusion_matrix = tf.Variable(tf.zeros([CLASSES,CLASSES]), name="confusion_matrix")
update_confusion_matrix = tf.assign_add(total_confusion_matrix, tf.matmul(tf.transpose(tf.one_hot(tf.argmax(network.outputs[TIME_DEPTH],1), CLASSES)), labels.outputs[TIME_DEPTH]))
reset_confusion_matrix = tf.assign(total_confusion_matrix, tf.zeros([CLASSES,CLASSES]))

update = tf.group(update_confusion_matrix, update_accloss, update_count)
reset = tf.group(reset_confusion_matrix, reset_accloss, reset_count)


if FLAGS.label_type == 'nhot':
  total_test_accuracy_labels = {}
  update_total_test_accuracy_labels = {}
  reset_total_test_accuracy_labels = {}
  average_accuracy_labels = {}
  for time in accuracy_labels.outputs:
    total_test_accuracy_labels[time] = tf.Variable(0.)
    update_total_test_accuracy_labels[time] = tf.assign_add(total_test_accuracy_labels[time], accuracy_labels.outputs[time])
    reset_total_test_accuracy_labels[time] = tf.assign(total_test_accuracy_labels[time], 0.)
    average_accuracy_labels[time] = total_test_accuracy_labels[time] / count

  # needed because tf 1.4 does not support grouping dict of tensor
  update_accloss = tf.stack((list(update_total_test_loss.values()) + list(update_total_test_accuracy.values()) +  list(update_total_test_accuracy_labels.values())))
  reset_accloss = tf.stack((list(reset_total_test_accuracy.values()) + list(reset_total_test_loss.values()) + list(reset_total_test_accuracy_labels.values())))

  update = tf.group(update_confusion_matrix, update_accloss, update_count)
  reset = tf.group(reset_confusion_matrix, reset_accloss, reset_count)


# decide which parameters get written to tfevents
# -----

with tf.name_scope('mean_test_time'):
  TEST_SUMMARIES.append(tf.summary.scalar('testtime_avg_cross_entropy', average_cross_entropy[TIME_DEPTH]))
  TEST_SUMMARIES.append(tf.summary.scalar('testtime_avg_accuracy', average_accuracy[TIME_DEPTH]))
  if FLAGS.label_type == 'nhot':
    TEST_SUMMARIES.append(tf.summary.scalar('testtime_avg_accuracy_labels', average_accuracy_labels[TIME_DEPTH]))

with tf.name_scope('accuracy_and_error'):
  TRAIN_SUMMARIES.append(tf.summary.scalar(error.name + "_{}".format(TIME_DEPTH), error.outputs[TIME_DEPTH]))
  TRAIN_SUMMARIES.append(tf.summary.scalar(accuracy.name + "_{}".format(TIME_DEPTH), accuracy.outputs[TIME_DEPTH]))
  if FLAGS.label_type == 'nhot':
    TRAIN_SUMMARIES.append(tf.summary.scalar(accuracy_labels.name + "_{}".format(TIME_DEPTH), accuracy_labels.outputs[TIME_DEPTH]))


#start session, merge summaries, start writers
#-----

with tf.Session() as sess:

  train_merged = tf.summary.merge(TRAIN_SUMMARIES)
  test_merged = tf.summary.merge(TEST_SUMMARIES)
 
  train_writer = tf.summary.FileWriter(RESULT_DIRECTORY + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(RESULT_DIRECTORY + '/validation')


  saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
  sess.run(tf.global_variables_initializer())

  # training and testing functions
  # -----

  def testing(train_it, data=dataset.validation, tag='Validation'):
    print(" " * 80 + "\r" + "[Validation]\tstarted", end="\r")
    sess.run([reset])
    while True:
      try:
        batch = data.next_batch(BATCH_SIZE)
        prepbatch = get_mnist.add_occluders(batch[0],BATCH_SIZE,FLAGS.n_occluders)
        
        _ = sess.run([update], feed_dict = {inp.placeholder: prepbatch , labels.placeholder: batch[label_index], is_training.placeholder: False, keep_prob.placeholder: 1.0})
      except (EOFError):
        break
    acc, loss, summary = sess.run([average_accuracy[TIME_DEPTH], average_cross_entropy[TIME_DEPTH], test_merged])
    print(" " * 80 + "\r" + "[{}]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}".format(tag, loss, acc, train_it))

    if not(FLAGS.restore_ckpt):
      test_writer.add_summary(summary, train_it)
    FLAGS.restore_ckpt = False
    return 0

  def training(train_it):
    while True:
      try:
        batch = dataset.train.next_batch(BATCH_SIZE)
        prepbatch = get_mnist.add_occluders(batch[0],BATCH_SIZE,FLAGS.n_occluders)
        summary, loss, acc, target, output = sess.run([train_merged, optimizer.outputs[TIME_DEPTH], accuracy.outputs[TIME_DEPTH], labels.outputs, network.outputs[TIME_DEPTH]], feed_dict = {inp.placeholder: prepbatch , labels.placeholder: batch[label_index], is_training.placeholder: True, keep_prob.placeholder:FLAGS.keep_prob})
        if (train_it % FLAGS.writeevery == 0):
          train_writer.add_summary(summary, train_it)
        if FLAGS.verbose:
          print(" " * 80 + "\r" + "[Training]\tloss: {:.5f}\tacc: {:.5f} \tstep: {}".format(loss, acc, train_it), end="\r")
        train_it = increment_global_step.eval()
      except (EOFError):
        _ = increment_global_epoch.eval()
        break
    return train_it



  # continueing from restored checkpoint
  # -----

  if FLAGS.restore_ckpt:
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIRECTORY)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print('[INFO]: Restored checkpoint successfully')
      # subtract epochs already done
      N_TRAIN_EPOCH -= global_epoch.eval()
      print('[INFO]: Continue training from last checkpoint: {} epochs remaining'.format(N_TRAIN_EPOCH))
      #sys.exit()
    else:
      print('[INFO]: No checkpoint found, starting experiment from scratch')
      FLAGS.restore_ckpt = False
      #sys.exit()


  # training loop
  # -----

  train_it = global_step.eval()
  for i_train_epoch in range(N_TRAIN_EPOCH):
    if (i_train_epoch % FLAGS.testevery == 0):
      _ = testing(train_it)
      saver.save(sess, CHECKPOINT_DIRECTORY + FLAGS.name + FLAGS.architecture \
      + FLAGS.dataset, global_step=train_it)
    train_it = training(train_it)


  # final test (ideally on an independent testset)
  # -----

  testing(train_it, data=dataset.test, tag='Testing')
  saver.save(sess, CHECKPOINT_DIRECTORY + FLAGS.name + FLAGS.architecture \
  + FLAGS.dataset, global_step=train_it)

  train_writer.close()
  test_writer.close()