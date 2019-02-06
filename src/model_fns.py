import numpy as np
import tensorflow as tf
from inception_v4 import inception_v4
from tensorflow.contrib.slim.nets import resnet_v2

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.slim import conv2d
from tensorflow.contrib.layers import xavier_initializer as xav_init


    
def normalize(images, axis=[2]):
    """ Modified from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py
    """
    num_pixels = math_ops.reduce_prod(array_ops.shape(images)[axis[0]]) # TODO multiple axis

    images = math_ops.cast(images, dtype=tf.float32)
    # E[X]
    image_mean = math_ops.reduce_mean(images, axis=axis, keepdims=True) # [?,?,1]

    # X^2
    squares = math_ops.square(images) # [?,?,pixels] 
    # E[X^2] - E[X]^2
    variance = (math_ops.reduce_mean(squares, axis=axis, keepdims=True) -
                math_ops.square(image_mean))
    # guarantee non-null (?)
    variance = gen_nn_ops.relu(variance)
    stddev = math_ops.sqrt(variance) # [?,?,1]

    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, tf.float32)) # [1]
    min_stddev = tf.fill(tf.shape(stddev), min_stddev) # [?,?,1]

    # adjust std dev, mean
    pixel_value_scale = math_ops.maximum(stddev, min_stddev)
    pixel_value_offset = image_mean

    # apply
    images = math_ops.subtract(images, pixel_value_offset)
    images = math_ops.div(images, pixel_value_scale)
    return images

# Helper fns for threesplit way of feeding
def _distance_squared(lts1, lts2): # Works for both simple vectors, or a batch of vectors
    difference = lts1 - lts2
    difference_sqrs = tf.square(difference)
    return tf.reduce_sum(difference_sqrs, -1)

def _get_threesplit_triplets(encodings, bucket_size, batch_size, embedding_size):
    """ 
    Encodings shape: (3*bucket_size, embedding_size)
    3*bucket_size = batch_size
    """
    assert 3*bucket_size == batch_size
    num_samples = batch_size
    
#     print(encodings.shape, bucket_size, num_samples, embedding_size)
    triplet_ids = tf.zeros((0,3), dtype=tf.int32)
    numbering = tf.constant(np.arange(num_samples), dtype=tf.int32)

    for _ in range(3): # three runs, every time another bucket can be anchor-positive pair
        # distances calculation between all pairs
        dists = tf.zeros([0,num_samples])
        for i in range(num_samples):
            distsi = tf.zeros([0])
            for j in range(num_samples):
                if i>j: 
                    distsi = tf.concat([distsi,dists[j,i:i+1]], axis=0)
                else: 
                    distij = _distance_squared(encodings[i], encodings[j])[None]
                    distsi = tf.concat([distsi, distij], axis=0)
            dists = tf.concat([dists, distsi[None,:]], axis=0)

        # for all anchor - positive pairs in first bucket, find semi-hard negative from other two buckets
        for i in range(bucket_size):
            oth_dists = dists[i,bucket_size:]
            for j in range(bucket_size):
                if i==j: continue
                pos_dist  = dists[i,j]

                # ignore values with distances smaller than anchor - positive distance
                oth_filtered = tf.where(
                                    oth_dists>pos_dist, 
                                    oth_dists,
                                    tf.constant(tf.float32.max, shape=oth_dists.shape)
                               )
                # from possible values, pick smallest
                negative = tf.argmin(oth_filtered, 0, output_type=tf.int32) + bucket_size

                first    = numbering[i]
                second   = numbering[j]
                negative = numbering[negative]
                cur_triplet_ids = tf.concat([first[None], second[None], negative[None]], axis=0)
                triplet_ids = tf.concat([triplet_ids, cur_triplet_ids[None]], axis=0)

        permut = np.array([(i+bucket_size)%num_samples for i in range(num_samples)])
        encodings = tf.gather(encodings, permut)
        numbering = tf.gather(numbering, permut)
    
#     return tf.reshape(triplet_ids, [-1, batch_size, 3])
    return triplet_ids


# Input feeding models
def _normal_model(images, model_fn, reuse, name, training, subtract_mean_pixel, normalize_input, **kwargs):
    if not name: name = 'normal_model'
    with tf.variable_scope(name, reuse=reuse):
        images = tf.cast(images, tf.float32)
        if subtract_mean_pixel:
            images = images - 234.766 # subtract mean pixel
        if normalize_input: # minus mean, divided through stddev
            images = normalize(images, axis=[1])
        images = tf.reshape(images, [-1, 224, 224, 1]) # image shape

        logits = model_fn(images, **kwargs)
        return logits
        
def _triplet_model(images, model_fn, reuse, name, training, subtract_mean_pixel, normalize_input, **kwargs):
    if not name: name = 'triplet_model'
    with tf.variable_scope(name, reuse=reuse):
        images = tf.cast(images, tf.float32)
        if subtract_mean_pixel:
            images = images - 234.766 # subtract mean pixel
        if normalize_input: # minus mean, divided through stddev
            images = normalize(images)
        images = tf.reshape(images, [-1, 3, 224, 224, 1]) # image shape

        logits1 = model_fn(images[:,0], **kwargs)
        logits2 = model_fn(images[:,1], **kwargs)
        logits3 = model_fn(images[:,2], **kwargs)

        return logits1, logits2, logits3
        
def _threesplit_model(images, model_fn, reuse, name, training, subtract_mean_pixel, normalize_input, 
                      embedding_size, batch_size, **kwargs):
    if not name: name = 'threesplit_model'
    with tf.variable_scope(name, reuse=reuse):
        images = tf.cast(images, tf.float32)
        if subtract_mean_pixel:
            images = images - 234.766 # subtract mean pixel
        if normalize_input: # minus mean, divided through stddev
            images = normalize(images, axis=[1])
        images = tf.reshape(images, [-1, 224, 224, 1]) # image shape

        logits = model_fn(images, embedding_size=embedding_size, **kwargs)
        triplet_permutations = _get_threesplit_triplets(logits, 
                                                        bucket_size=batch_size//3, 
                                                        batch_size=batch_size, 
                                                        embedding_size=embedding_size)
        logitsAAB = tf.gather(logits, tf.reshape(triplet_permutations, [-1]))
        logitsAAB = tf.reshape(logitsAAB, [-1, 3, embedding_size]) # 270 x 3 x 64 --> 2nd dimension classes AAB
        
        print(embedding_size)
        return logitsAAB[:,0], logitsAAB[:,1], logitsAAB[:,2]

def _vgg16(inputs, training=True, embedding_size=64, 
           dropout_keep_prob=0.5,
           middleRepr=False,
           scope='vgg_16'):
    """ From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
    
Oxford Net VGG 16-Layers version D Example without fcn/convolutional layers at end

  """
    with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
        
        net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
        
        net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
        
        net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
        
        net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
        
        net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')

        if middleRepr:
            return net

        net = tf.layers.dense(net, embedding_size, 
                              activation=tf.nn.relu, kernel_initializer=xav_init(), name='fc8')
            
        return net   
        
def _inceptionv4(input_img, training=True, embedding_size=64, dropout_keep_prob=0.5, middleRepr=False):
    with tf.variable_scope('inceptionv4', reuse=tf.AUTO_REUSE):
        logits, _ = inception_v4(input_img, num_classes=None, is_training=training)

        if middleRepr:
            return logits

        logits = conv2d(logits, embedding_size, [1, 1], activation_fn=None,
                        normalizer_fn=None, scope='logits')
        logits = tf.squeeze(logits, [1, 2])
        return logits
    
def _resnet50(input_img, training=True, embedding_size=64, dropout_keep_prob=0.5, middleRepr=False):
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
        logits, _ = resnet_v2.resnet_v2_50(input_img, is_training=training) # ?x1x1x2048
        
        if middleRepr:
            return tf.squeeze(logits, [1,2])

        logits = conv2d(logits, embedding_size, [1, 1], activation_fn=None,
                        normalizer_fn=None, scope='logits')
        logits = tf.squeeze(logits, [1, 2])
        return logits
    
def _model(model_fn, images, mode='triplets', name=None, reuse=tf.AUTO_REUSE, 
          subtract_mean_pixel=False, normalize_input=True, # input normalization params
          middleRepr=False, embedding_size=64,        # net params
          batch_size=30, training=True): # required for threesplit
    if not mode:
        return _normal_model(images, model_fn, reuse, name, training,
                       subtract_mean_pixel, normalize_input,                           # input normalization params
                       middleRepr=middleRepr, embedding_size=embedding_size  # conv net params
        )
    elif mode == 'triplets': 
        return _triplet_model(images, model_fn, reuse, name, training,
                       subtract_mean_pixel, normalize_input,                           # input normalization params
                       middleRepr=middleRepr, embedding_size=embedding_size  # conv net params
        )
    elif mode == 'threesplit':
        return _threesplit_model(images, model_fn, reuse, name, training,
                       subtract_mean_pixel, normalize_input,                           # input normalization params
                       middleRepr=middleRepr, embedding_size=embedding_size, # conv net params
                       batch_size=batch_size
        )
    else: raise NotImplementedError
    
    
def vgg16(*args, **kwargs):      
    return _model(_vgg16, *args, **kwargs)
        
def inceptionv4(*args, **kwargs):      
    return _model(_inceptionv4, *args, **kwargs)
        
def resnet50(*args, **kwargs):
    return _model(_resnet50, *args, **kwargs)

def _fcn(logits, logits_out, name, activation=tf.nn.relu,
         training=True, trainable=True, bn_before=False, bn_after=False):
    if bn_before: logits = tf.layers.batch_normalization(logits,  
                                                         training=training, trainable=trainable,
                                                         name=name + '_bn_before')
    logits = tf.layers.dense(logits, logits_out, activation=activation, name=name)
    if bn_after:  logits = tf.layers.batch_normalization(logits,
                                                         training=training, trainable=trainable,
                                                         name=name + '_bn_after')
    return logits

def classification(logits, layer_sizes=[64,10], name='classifier',
                   bnorm_before=False, bnorm_middle=False, training=True, trainable=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i,layer_size in enumerate(layer_sizes):
            layer_name = 'layer' + str(i+1)
            
            if i==0 and bnorm_before:
                logits = _fcn(logits, layer_size, layer_name,
                              training=training, trainable=trainable, bn_before=True, bn_after=bnorm_middle)
            elif (i+1) == len(layer_sizes): # last layer
                return _fcn(logits, layer_size, layer_name, activation=None,
                            trainable=trainable, training=training) # No BatchNormalization
            else:
                logits = _fcn(logits, layer_size, layer_name,
                              training=training, trainable=trainable, bn_after=bnorm_middle)
        

class ConvNet:
    def __init__(self, model='resnet', suffix='',
                 feed_mode='threesplit', 
                 loss_mode='hinge', loss_mode_params = {'alpha': 1},
                 embedding_size=64,
                 batch_size=30,
                 middleRepr=False,
                 learning_rate=1e-4, # Required to have different names for differently trained models
                 annealing=0,        # dito
                 small=True,
                 scope='cnn'):
        self.model_name = '{}_{}_{}_es{}'.format(feed_mode, model, loss_mode, embedding_size)
        if middleRepr:
            self.model_name += '_middleRepr'
        if learning_rate != 1e-4:
            self.model_name += '_lr' + str(learning_rate)
        if annealing:
            self.model_name += '_ann-' + str(annealing)
        if not small:
            self.model_name += '_fullModel'
        if suffix:
            self.model_name += '_' + suffix

        self.feed_mode  = feed_mode
        self.loss_mode  = loss_mode
        self.loss_mode_params = loss_mode_params
        self.embedding_size = embedding_size
        self.batch_size = batch_size # for threesplit model
        self.middleRepr = middleRepr
        self.scope = scope
        
        # CNN model implementation
        self.model_fn = None
        if model == 'resnet':
            self.model_fn = resnet50
        elif model == 'inception':
            self.model_fn = inceptionv4
        elif model == 'vgg':
            self.model_fn = vgg16
        else:
            raise Exception("Model '{}' not recognized".format(model))
                        
    def logits(self, images, training=True, feed_mode=None, stop_gradient=False):
        # Choose name based on initial feed_mode to share weights of CNN
        # But also allow calculations in different feed_modes
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            logits = self.model_fn(images, training=training, mode=feed_mode, embedding_size=self.embedding_size,
                                   name=self.feed_mode, middleRepr=self.middleRepr, batch_size=self.batch_size)
            if stop_gradient:
                logits = tf.stop_gradient(logits)
            return logits
   
    def restore(self, sess, checkpoint_dir='../checkpts/', checkpoint_num=50000):
        restorer = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        )
        checkpoint_dir += self.model_name
        restorer.restore(sess, checkpoint_dir + '/checkpoint-' + str(checkpoint_num))
