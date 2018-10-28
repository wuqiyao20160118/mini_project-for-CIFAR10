import tensorflow as tf
import numpy as np
import math
import timeit
import time
import datetime
import keras
import matplotlib.pyplot as plt
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
MOMENTUM=0.9

def resnet_block(name,inputs,in_filters,num_filters=16,
                  kernel_size=3,strides=[1,1,1,1],
                  activation='relu',is_training=False):
    x = conv(name,inputs,kernel_size,in_filters,num_filters,strides)
    x = batch_norm(name,x,is_training)
    if(activation):
        x = tf.nn.relu(x)
    return x

def conv(name,inputs,kernel_size, in_filters, num_filters, strides):
    # Note: the input size should be:[batch, in_height, in_width, in_channels]
    # the filter size should be: [filter_height, filter_width, in_channels, out_channels]
    with tf.variable_scope(name):
        n = kernel_size * kernel_size * num_filters
      # random initialization
        kernel = tf.get_variable('DW', [kernel_size, kernel_size, in_filters, num_filters],tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(inputs, kernel, strides, padding='SAME')


def batch_norm(name,x,is_training):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x,training=is_training)

def global_avg_pool(x):
    pool1=tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=1)
    return pool1

def resnet(x,input_channel=3,is_training=False):
    with tf.variable_scope('unit_1'):
        x=resnet_block(name='unit_1_0',inputs=x,in_filters=input_channel,is_training=is_training)
    for i in range(6):
        with tf.variable_scope('unit_1_%d' % i):
            a = resnet_block(name='unit_1_%d_0' % i,inputs=x,in_filters=16,is_training=is_training)
            b = resnet_block(name='unit_1_%d_1' % i,inputs=a,in_filters=16,activation=None,is_training=is_training)
            x = keras.layers.add([x,b])
            x = tf.nn.relu(x)
    # out：32*32*16
    with tf.variable_scope('unit_2'):
        for i in range(6):
            with tf.variable_scope('unit_2_%d' % i):
                if i==0:
                    a = resnet_block(name='unit_2_%d_0' % i,inputs=x,strides=[1,2,2,1],in_filters=16,num_filters=32,is_training=is_training)
                else:
                    a = resnet_block(name='unit_2_%d_0' % i,inputs=x,strides=[1,1,1,1],in_filters=32,num_filters=32,is_training=is_training)
                b = resnet_block(name='unit_2_%d_1' % i,inputs=a,in_filters=32,num_filters=32,activation=None,is_training=is_training)
                if i==0:
                    x = conv(name='unit_2_0_2',inputs=x,kernel_size=3,in_filters=16,num_filters=32,strides=[1,2,2,1])
            x = keras.layers.add([x,b])
            x = tf.nn.relu(x)
    # out:16*16*32
    with tf.variable_scope('unit_3'):
        for i in range(6):
            with tf.variable_scope('unit_3_%d' % i):
                if i==0:
                    a = resnet_block(name='unit_3_%d_0' % i,inputs=x,strides=[1,2,2,1],in_filters=32,num_filters=64,is_training=is_training)
                else:
                    a = resnet_block(name='unit_3_%d_0' % i,inputs=x,strides=[1,1,1,1],in_filters=64,num_filters=64,is_training=is_training)
                b = resnet_block(name='unit_3_%d_1' % i,inputs=a,in_filters=64,num_filters=64,activation=None,is_training=is_training)
                if i==0:
                    x = conv(name='unit_3_0_2',inputs=x,kernel_size=3,in_filters=32,num_filters=64,strides=[1,2,2,1])
                x = keras.layers.add([x,b])
                x = tf.nn.relu(x)
    # out:8*8*64
    x=global_avg_pool(x)
    # out:7*7*64
    with tf.variable_scope('logit'):
        x = tf.reshape(x,[-1,7*7*64])
        x = tf.layers.dense(inputs=x, units=10,use_bias=True,activation=None)
    return x