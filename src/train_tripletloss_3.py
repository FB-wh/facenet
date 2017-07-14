# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import math
import json
import random
from nets import inception_v2
from nets.inception_v3 import inception_v3_arg_scope

import faiss
from guuker import prt

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from deployment import model_deploy
slim = tf.contrib.slim

SEED = 727
class AutoDataset:
  def __init__(self):
    #json_input = '/raid5data/dplearn/taobao/crawler_tbimg/out/autohome.json'
    json_input = '../data.json'
    image_root = '/raid5data/dplearn/taobao/crawler_tbimg/auto_images'
    self.data_map = {}
    self.key_list = []
    with open(json_input, 'r') as f:
      for line in f:
        data = json.loads(line)
        _type = data['type']
        if _type!='image':
          continue
        car_id = data['car_id']
        pos = data['pos']
        jpg_path = "%s/%d_%d.jpg" % (image_root, car_id, pos)
        #with open(jpg_path, 'rb') as fb:
        #  encoded_jpg = fb.read()
        if not car_id in self.data_map:
          self.data_map[car_id] = []
          self.key_list.append(car_id)
        #self.data_map[car_id].append(encoded_jpg)
        self.data_map[car_id].append(jpg_path)

  def split_train_val(self, train_proportion, seed):
    random.Random(seed).shuffle(self.key_list)
    train_count = int(len(self.key_list)*train_proportion)
    self.train_key_list = self.key_list[:train_count]
    self.val_key_list = self.key_list[train_count:]

  #def train_key_list():
  #  return self.train_key_list

  #def val_key_list():
  #  return self.val_key_list

  def get_contents(self, car_id):
    return self.data_map[car_id]

NUM_GPUS = 0
dataset = AutoDataset()
dataset.split_train_val(0.7, SEED)
prt("dataset loaded %d %d" % ( len(dataset.train_key_list),  len(dataset.val_key_list) ))
BASE_EPOCH_SIZE = 10000

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='./logs')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='./models')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.99)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', default='./inception_v2.ckpt')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--entities_per_batch', type=int,
        help='Number of entities per batch.', default=54)
    parser.add_argument('--images_per_entity', type=int,
        help='Number of images per entity.', default=20)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=BASE_EPOCH_SIZE//NUM_GPUS)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--dropout_keep_prob', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=SEED)
    parser.add_argument('--num_gpus', type=int,
        help='Num of used gpus.', default=NUM_GPUS)
    parser.add_argument('--select_level', type=int,
        help='triplets select level.', default=1)

    return parser.parse_args(argv)

NUM_GPUS = 0
cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
for de in cuda_devices.split(','):
  de = de.strip()
  if len(de)==0:
    continue
  NUM_GPUS += 1

if NUM_GPUS==0:
  print('no gpu set, exit')
  sys.exit(1)
args = parse_arguments(sys.argv[1:])



def image_to_embedding(inputs, is_training, args):
  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
    with slim.arg_scope(inception_v3_arg_scope(weight_decay=args.weight_decay)):
      logits, _ = inception_v2.inception_v2(inputs, num_classes = args.embedding_size, is_training = is_training, dropout_keep_prob = args.dropout_keep_prob, last_pool_size = int(args.image_size*7/224))
      embeddings = tf.nn.l2_normalize(logits, 1, 1e-10, name='embeddings')
  return embeddings

def get_learning_rate(args):
  global_step = variables.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
      args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)
  return learning_rate

def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def get_optimizer(args, learning_rate):
  optimizer = args.optimizer
  if optimizer=='ADAGRAD':
      opt = tf.train.AdagradOptimizer(learning_rate)
  elif optimizer=='ADADELTA':
      opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
  elif optimizer=='ADAM':
      opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
  elif optimizer=='RMSPROP':
      opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
  elif optimizer=='MOM':
      opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
  else:
      raise ValueError('Invalid optimization algorithm')
  return opt

def get_train_op(args, total_loss, learning_rate):
    # Generate moving averages of all losses and associated summaries.
    global_step = variables.get_or_create_global_step()
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
    
        opt = get_optimizer(args, learning_rate)
        grads = opt.compute_gradients(total_loss, tf.global_variables())
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(args.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op

def triplet_loss_fn(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss

def main():
    print(args)
    prt('')
  

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))

    np.random.seed(seed=args.seed)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(num_clones=args.num_gpus,clone_on_cpu=False)
        tf.set_random_seed(args.seed)
        #global_step = tf.Variable(0, trainable=False)
        global_step = variables.get_or_create_global_step()

        # Placeholder for the learning rate
        #learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        #batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        with tf.device('/cpu:0'):
          is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
          image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
          labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
          
          input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                      dtypes=[tf.string, tf.int64],
                                      shapes=[(3,), (3,)],
                                      shared_name=None, name=None)
          enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
          
          nrof_preprocess_threads = 8
          images_and_labels = []
          for _ in range(nrof_preprocess_threads):
              filenames, label = input_queue.dequeue()
              #filenames = tf.Print(filenames, [tf.shape(filenames)], 'filenames shape:')
              images = []
              for filename in tf.unstack(filenames):
                  #filename = tf.Print(filename, [filename], 'filename = ')
                  file_contents = tf.read_file(filename)
                  image = tf.image.decode_jpeg(file_contents)
                  #image = tf.Print(image, [tf.shape(image)], 'data count = ')
                  if image.dtype != tf.float32:
                    image= tf.image.convert_image_dtype(image, dtype=tf.float32)
                  if args.random_crop:
                      #image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                      bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
                      sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                          tf.shape(image),
                          bounding_boxes=bbox,
                          area_range=(0.7,1.0),
                          use_image_if_no_bounding_boxes=True)
                      bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
                      image = tf.slice(image, bbox_begin, bbox_size)
                  #else:
                  #    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                  image = tf.expand_dims(image, 0)
                  image = tf.image.resize_bilinear(image, [args.image_size, args.image_size], align_corners=False)
                  image = tf.squeeze(image, [0])
                  if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
                  image.set_shape((args.image_size, args.image_size, 3))
                  ##pylint: disable=no-member
                  image = tf.subtract(image, 0.5)
                  image = tf.multiply(image, 2.0)
                  #image = tf.Print(image, [tf.shape(image)], 'data count = ')
                  images.append(image)
                  #images.append(tf.image.per_image_standardization(image))
              images_and_labels.append([images, label])
      

          learning_rate = get_learning_rate(args)
          opt = get_optimizer(args, learning_rate)
          image_batch, label_batch = tf.train.batch_join(
              images_and_labels, batch_size=args.batch_size, 
              shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
              capacity=4 * nrof_preprocess_threads * args.batch_size,
              allow_smaller_final_batch=False)
          batch_queue = slim.prefetch_queue.prefetch_queue(
              [image_batch, label_batch], capacity=9000)

        def clone_fn(_batch_queue):
          _image_batch, _label_batch = _batch_queue.dequeue()
          embeddings = image_to_embedding(_image_batch, is_training_placeholder, args)

          # Split embeddings into anchor, positive and negative and calculate triplet loss
          anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
          triplet_loss = triplet_loss_fn(anchor, positive, negative, args.alpha)
          tf.losses.add_loss(triplet_loss)
          #tf.summary.scalar('learning_rate', learning_rate)
          return embeddings, _label_batch, triplet_loss

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone = clones[0]
        triplet_loss = first_clone.outputs[2]
        embeddings = first_clone.outputs[0]
        _label_batch = first_clone.outputs[1]
        #embedding_clones = model_deploy.create_clones(deploy_config, embedding_fn, [batch_queue])

        #first_clone_scope = deploy_config.clone_scope(0)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        update_ops = []
        vdic = [v for v in tf.trainable_variables() if v.name.find("Logits/")<0]
        pretrained_saver = tf.train.Saver(vdic)
        saver = tf.train.Saver(max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        with tf.device(deploy_config.optimizer_device()):
          learning_rate = get_learning_rate(args)
          opt = get_optimizer(args, learning_rate)

        total_loss, clones_gradients= model_deploy.optimize_clones(
            clones,
            opt,
            var_list= tf.trainable_variables())

        grad_updates= opt.apply_gradients(clones_gradients, global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        
        vdic = [v for v in tf.trainable_variables() if v.name.find("Logits/")<0]
        pretrained_saver = tf.train.Saver(vdic)
        saver = tf.train.Saver(max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        
        sess = tf.Session()

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={is_training_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={is_training_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                pretrained_saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                eval_one_epoch(args, sess, dataset, image_paths_placeholder, labels_placeholder,
                    is_training_placeholder, enqueue_op, clones)
                # Train for one epoch
                train_one_epoch(args, sess, dataset, image_paths_placeholder, labels_placeholder,
                    is_training_placeholder, enqueue_op, input_queue, 
                    clones, total_loss, train_op, summary_op, summary_writer)

                # Save variables and the metagraph if it doesn't exist already
                global_step = variables.get_or_create_global_step()
                step = sess.run(global_step, feed_dict=None)
                print('one epoch finish', step)
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
                print('saver finish')


    sess.close()
    return model_dir


def train_one_epoch(args, sess, dataset, image_paths_placeholder, labels_placeholder,
          is_training_placeholder, enqueue_op, input_queue,
          clones, loss, train_op, summary_op, summary_writer):
    global_step = variables.get_or_create_global_step()
    step = sess.run(global_step, feed_dict=None)
    epoch = step // args.epoch_size
    batch_number = 0
    
    lr = args.learning_rate
    batch_size = args.batch_size * args.num_gpus
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        prt('start to sample entities')
        image_paths, num_per_class = sample_entities(args, dataset)
        #print(num_per_class[0:5])
        #prt(len(image_paths))
        #print(num_per_class)
        #print(image_paths[0:10])
        
        #print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = len(image_paths)
        assert(nrof_examples%batch_size==0)
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
        #print(image_paths_array.shape)
        #print(labels_array.shape)
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, args.embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        embeddings = clones[0].outputs[0]
        label_batch = clones[0].outputs[1]
        #print(nrof_batches)
        for i in xrange(nrof_batches):
          emb, lab = sess.run([embeddings, label_batch], feed_dict={is_training_placeholder: True})
          emb_array[lab,:] = emb
        print('time for fetching all embedding %.3f' % (time.time()-start_time))
        #print(emb_array[0:5,0:5])

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, triplets_info = select_triplets(args, emb_array, num_per_class, image_paths)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (0, len(triplets), selection_time))

        assert len(triplets)>0
        #post-processing
        assert (args.batch_size%3==0)
        triplet_size = batch_size // 3
        _a = len(triplets) // triplet_size
        nrof_triplets = _a * triplet_size
        triplets = triplets[0:nrof_triplets]
        #post-processing finish

        # Perform training on the selected triplets
        triplet_paths = list(itertools.chain(*triplets))
        nrof_batches = int(np.ceil(nrof_triplets*3/batch_size))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, args.embedding_size))
        #loss_array = np.zeros((nrof_triplets,))
        prt('nrof_batches: %d' % nrof_batches)
        while i < nrof_batches:
            start_time = time.time()

            #err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict={is_training_placeholder:True})
            #emb_array[lab,:] = emb
            #loss_array[i] = err

            err, _, step = sess.run([loss, train_op, global_step], feed_dict={is_training_placeholder:True})
            duration = time.time() - start_time
            prt('Epoch: [%d][%d@%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, i, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration

        prt('one sample finish')
        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step
  
def select_triplets_1(args, embeddings, nrof_images_per_class, image_paths):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    print('selecting triplets', len(nrof_images_per_class))

    for i in xrange(args.entities_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<args.alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips

def select_triplets_2(args, embeddings, nrof_images_per_class, image_paths):
    # semi-hard mining
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    
    print('selecting triplets via semi-hard', len(nrof_images_per_class))

    for i in xrange(args.entities_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<args.alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                #all_neg = np.where(neg_dists_sqr-pos_dist_sqr<args.alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips

def select_triplets(args, embeddings, nrof_images_per_class, image_paths):
  if args.select_level==1:
    return select_triplets_1(args, embeddings, nrof_images_per_class, image_paths)
  elif args.select_level==2:
    return select_triplets_2(args, embeddings, nrof_images_per_class, image_paths)
  else:
    return [], None

def sample_entities(args, dataset):
  nrof_images = args.entities_per_batch * args.images_per_entity
  train_key_list = dataset.train_key_list

  # Sample classes from the dataset
  nrof_classes = len(train_key_list)
  class_indices = np.arange(nrof_classes)
  np.random.shuffle(class_indices)
  
  i = 0
  image_paths = []
  num_per_class = []
  sampled_class_indices = []
  # Sample images from these classes until we have enough
  while len(image_paths)<nrof_images:
    class_index = class_indices[i]
    eid = train_key_list[class_index]
    contents = dataset.get_contents(eid)
    nrof_images_in_class = len(contents)
    image_indices = np.arange(nrof_images_in_class)
    np.random.shuffle(image_indices)
    nrof_images_from_class = min(nrof_images_in_class, args.images_per_entity, nrof_images-len(image_paths))
    idx = image_indices[0:nrof_images_from_class]
    image_paths_for_class = [contents[j] for j in idx]
    sampled_class_indices += [class_index]*nrof_images_from_class
    image_paths += image_paths_for_class
    num_per_class.append(nrof_images_from_class)
    i+=1

  return image_paths, num_per_class

def all_val_entities(args, dataset):
  val_key_list = dataset.val_key_list

  image_paths = []
  num_per_class = []
  for i in xrange(len(val_key_list)):
    key = val_key_list[i]
    contents = dataset.get_contents(key)
    if len(contents)==0:
      continue
    image_paths += contents
    num_per_class.append(len(contents))
  return image_paths, num_per_class

def top1_recall(emb_array, num_per_class):
  print(emb_array.shape)
  labels = np.array( [0]*emb_array.shape[0], dtype=np.int )
  pos_beg = 0
  label = 1
  for num in num_per_class:
    pos_end = pos_beg+num
    labels[pos_beg:pos_end] = label
    pos_beg = pos_end
    label += 1
  d = emb_array.shape[1]
  quantizer = faiss.IndexFlatL2(d)  # the other index
  index = faiss.IndexIVFFlat(quantizer, d, 10, faiss.METRIC_L2)
  assert not index.is_trained
  index.train(emb_array)
  index.add(emb_array)
  assert index.is_trained
  index.nprobe = 3
  k = 2
  D, I = index.search(emb_array, k)     # actual search
  correct = 0
  for i in xrange(I.shape[0]):
    #assert(I[i][0]==i)
    for j in xrange(k):
      idx = I[i][j]
      if idx!=i:
        label1 = labels[i]
        label2 = labels[idx]
        if label1==label2:
          correct+=1
        break
    #print(i, I[i], correct)
  r = correct/I.shape[0]
  return r

def eval_one_epoch(args, sess, dataset, image_paths_placeholder, labels_placeholder,
    is_training_placeholder, enqueue_op, clones):
  batch_size = args.batch_size*args.num_gpus
  image_paths, num_per_class = all_val_entities(args, dataset)
  print('eval image paths',len(image_paths))
  nrof_origin_samples = len(image_paths)
  assert(sum(num_per_class)==nrof_origin_samples)
  #print(num_per_class)
  #print(image_paths[0:10])
  assert (args.batch_size%3==0)
  triplet_size = args.batch_size // 3
  _a = int(math.ceil(len(image_paths) / batch_size))
  nrof_samples = _a * batch_size
  while nrof_samples>len(image_paths):
    image_paths.append(image_paths[0])

  #for p in image_paths:
  #  print(p)
  
  #print('Running forward pass on sampled images: ', end='')
  start_time = time.time()
  nrof_examples = len(image_paths)
  assert(nrof_examples%batch_size==0)
  labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
  image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
  print(image_paths_array.shape)
  print(labels_array.shape)
  sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
  emb_array = np.zeros((nrof_examples, args.embedding_size), dtype=np.float32)
  nrof_batches = int(np.ceil(nrof_examples / batch_size))
  print('eval batches', nrof_batches)
  for i in xrange(nrof_batches):
    if i%10==0:
      prt('running eval batch %d'%i)
    ops = []
    for clone in clones:
      with tf.device(clone.device):
        embeddings, labels, _ = clone.outputs
      ops += [embeddings, labels]
    ops_value = sess.run(ops, feed_dict={is_training_placeholder: False})
    for k in xrange(args.num_gpus):
      emb = ops_value[k*2]
      #prt(emb.shape)
      lab = ops_value[k*2+1]
      #prt(lab.shape)
      emb_array[lab,:] = emb
    sys.stdout.flush()
  print('%.3f' % (time.time()-start_time))
  emb_array = emb_array[0:nrof_origin_samples,:]
  score = top1_recall(emb_array, num_per_class)
  print('top1 recall: %f' % score)



def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

  

if __name__ == '__main__':
    main()


