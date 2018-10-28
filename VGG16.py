import tensorflow as tf
import numpy as np
import math
import timeit
import time
import datetime
import matplotlib.pyplot as plt
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

def max_pool(x,p_size=2,s_size=2):
    return tf.nn.max_pool(x,ksize=[1,p_size,p_size,1],strides=[1,s_size,s_size,1],padding='VALID')

def leaky_relu(x,alpha=0.01):
    return tf.maximum(x,alpha*x)

def batch_norm(name,x,is_training):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x,center=False,scale=False,training=is_training)

def conv(name,x,filter_size, in_filters, out_filters, strides):
    # Note: the input size should be:[batch, in_height, in_width, in_channels]
    # the filter size should be: [filter_height, filter_width, in_channels, out_channels] (H,W,C_in,C_out)
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
      # random initialization
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

def batch_norm(name,x,is_training):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x,center=False,scale=False,training=is_training)

def build_model(images,is_training=False):
    with tf.variable_scope('block1'):
        x=images
        x=conv("conv1_1",x,3,3,64,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn1_1',x,is_training)
        x=conv("conv1_2",x,3,64,64,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn1_2',x,is_training)
        tf.logging.info('image after unit %s', x.get_shape())
    with tf.variable_scope('block2'):
        x=conv("conv2_1",x,3,64,128,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn2_1',x,is_training)
        x=conv("conv2_2",x,3,128,128,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn2_2',x,is_training)
        tf.logging.info('image after unit %s', x.get_shape())
    with tf.variable_scope('block3'):
        x=conv("conv3_1",x,3,128,256,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn3_1',x,is_training)
        x=conv("conv3_2",x,3,256,256,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn3_2',x,is_training)
        x=conv("conv3_3",x,1,256,256,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn3_3',x,is_training)
        x=max_pool(x)
        tf.logging.info('image after unit %s', x.get_shape())
    with tf.variable_scope('block4'):
        x=conv("conv4_1",x,3,256,512,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn4_1',x,is_training)
        x=conv("conv4_2",x,3,512,512,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn4_2',x,is_training)
        x=conv("conv4_3",x,1,512,512,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn4_3',x,is_training)
        tf.logging.info('image after unit %s', x.get_shape())
    with tf.variable_scope('block4'):
        x=conv("conv5_1",x,3,512,512,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn5_1',x,is_training)
        x=conv("conv5_2",x,3,512,512,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn5_2',x,is_training)
        x=conv("conv5_3",x,1,512,512,[1,1,1,1])
        x=leaky_relu(x)
        x=batch_norm('bn5_3',x,is_training)
        x=max_pool(x)
        tf.logging.info('image after unit %s', x.get_shape())
    with tf.variable_scope('fc'):
        x = tf.reshape(x,[-1,8*8*512])
        x = tf.layers.dense(inputs=x, units=64,use_bias=True,activation=None)
        x=leaky_relu(x)
        x=batch_norm('bn6',x,is_training)
        x = tf.layers.dropout(x,training=is_training)
        x = tf.layers.dense(inputs=x, units=64,use_bias=True,activation=None)
        x=leaky_relu(x)
        x=batch_norm('bn7',x,is_training)
        x = tf.layers.dropout(x,training=is_training) #add dropout layer
        x = tf.layers.dense(inputs=x, units=10,use_bias=True,activation=None)
        tf.logging.info('image after unit %s', x.get_shape())
        return x

