import tensorflow as tf
import numpy as np
import math
import timeit
import time
import datetime
import matplotlib.pyplot as plt
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
MOMENTUM=0.9

def resnet_block(name,inputs,in_filters,num_filters=16,
                  kernel_size=3,strides=[1,1,1,1],
                  activation='relu',is_training):
    x = conv(name,inputs,kernel_size,in_filters,num_filters,strides)
    x = batch_norm(name,x,is_training)
    if(activation):
        x = tf.nn.relu(x)
    return x


def max_pool(x,p_size=3,s_size=2):
    return tf.nn.max_pool(x,ksize=[1,p_size,p_size,1],strides=[1,s_size,s_size,1],padding='VALID')

def leaky_relu(x,alpha=0.01):
    return tf.maximum(x,alpha*x)

def build_model(images,y,use_bottleneck=False,is_training=False):
    with tf.variable_scope('init'):
        x = images
        """the first conv layer（3,3x3/1,16）"""
        x = conv('init_conv', x, 3, 3, 16, [1,1,1,1])
        print(x.get_shape())

    # 残差网络参数
    strides = [1, 2, 2]
    # 激活前置
    activate_before_residual = [True, False, False]
    if use_bottleneck:
      # bottleneck残差单元模块
      res_func = False
      # 通道数量
      filters = [16, 64, 128, 256]
    else:
      # 标准残差单元模块
      res_func = True
      # 通道数量
      filters = [16, 16, 32, 64]

    # 第一组
    with tf.variable_scope('unit_1_0'):
        x = _switchFunction(res_func,x, filters[0], filters[1], 
                   [1,1,1,1],
                   activate_before_residual[0],is_training)
    for i in range(1, 5):
        with tf.variable_scope('unit_1_%d' % i):
            x = _switchFunction(res_func,x, filters[1], filters[1], [1,1,1,1], False,is_training)
            print(x.get_shape())

    # 第二组
    with tf.variable_scope('unit_2_0'):
        x = _switchFunction(res_func,x, filters[1], filters[2], 
                   [1,2,2,1],
                   activate_before_residual[1],is_training)
    for i in range(1, 5):
        with tf.variable_scope('unit_2_%d' % i):
            x = _switchFunction(res_func,x, filters[2], filters[2], [1,1,1,1], False,is_training)
            print(x.get_shape())
        
    # 第三组
    with tf.variable_scope('unit_3_0'):
        x = _switchFunction(res_func,x, filters[2], filters[3], [1,2,2,1],
                   activate_before_residual[2],is_training)
    for i in range(1, 5):
        with tf.variable_scope('unit_3_%d' % i):
            x = _switchFunction(res_func,x, filters[3], filters[3], [1,1,1,1], False,is_training)
            print(x.get_shape())

    # average pooling layer
    with tf.variable_scope('unit_last'):
        x = batch_norm('final_bn', x,is_training)
        x=leaky_relu(x)
        x = global_avg_pool(x)
        print(x.get_shape())
    with tf.variable_scope('logit'):
        x = tf.reshape(x,[-1,7*7*64])
        x = tf.layers.dense(inputs=x, units=1000,use_bias=True,activation=tf.nn.relu)
        x = batch_norm('final_bn1', x,is_training)
        x = tf.layers.dropout(x,training=is_training)
        x = tf.layers.dense(inputs=x, units=10,use_bias=True,activation=None)
        print(x.get_shape())
        #tf.logging.debug('image after unit 1', x.get_shape())
    return x

def _switchFunction(res_function,x, in_filter, out_filter, stride, activate_before_residual=False,is_training=False):
    if res_function:
        x=_bottleneck_residual(x, in_filter, out_filter, stride,activate_before_residual,is_training)
    else:
        x=_residual(x, in_filter, out_filter, stride, activate_before_residual,is_training)
    return x

def _residual(x, in_filter, out_filter, stride, activate_before_residual=False,is_training=False):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            # 先做BN和ReLU激活
            x = batch_norm('init_bn', x,is_training)
            # 获取残差直连
            x=leaky_relu(x)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            # 获取残差直连
            orig_x = x
            # 后做BN和ReLU激活
            x = batch_norm('init_bn', x,is_training)
            x=leaky_relu(x)

    # 第1子层
    with tf.variable_scope('sub1'):
        # 3x3卷积，使用输入步长，通道数(in_filter -> out_filter)
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
        # BN和ReLU激活
        x = batch_norm('bn2', x,is_training)
        x=leaky_relu(x)
        # 3x3卷积，步长为1，通道数不变(out_filter)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
    # 合并残差层
    with tf.variable_scope('sub_add'):
        # 当通道数有变化时
        if in_filter != out_filter:
            # 均值池化，无补零
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            # 补零(第4维前后对称补零)
            orig_x = tf.pad(orig_x, 
                        [[0, 0], 
                         [0, 0], 
                         [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]
                        ])
    # 合并残差
        x += orig_x

    #tf.logging.debug('image after unit %s', x.get_shape())
    return x

 # bottleneck残差单元模块
def _bottleneck_residual(x, in_filter, out_filter, stride,activate_before_residual=False,is_training=False):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            # 先做BN和ReLU激活
            x = batch_norm('init_bn', x,is_training)
            x=leaky_relu(x)
            # 获取残差直连
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            # 获取残差直连
            orig_x = x
            # 后做BN和ReLU激活
            x = batch_norm('init_bn', x,is_training)
            x=leaky_relu(x)
    # 第1子层
    with tf.variable_scope('sub1'):
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
        x = conv('conv1', x, 1, in_filter, out_filter/4, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
        # BN和ReLU激活
        x = batch_norm('bn2', x,is_training)
        x=leaky_relu(x)
        # 3x3卷积，步长为1，通道数不变(out_filter/4)
        x = conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    # 第3子层
    with tf.variable_scope('sub3'):
        # BN和ReLU激活
        x = batch_norm('bn3', x,is_training)
        x=leaky_relu(x)
        # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
        x = conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    # 合并残差层
    with tf.variable_scope('sub_add'):
        # 当通道数有变化时
        if in_filter != out_filter:
            # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
            orig_x = conv('project', orig_x, 1, in_filter, out_filter, stride)
      
      # 合并残差
        x += orig_x

    #tf.logging.info('image after unit %s', x.get_shape())
    return x

def decay():
    costs = []
    for var in tf.trainable_variables():
      #calculate all variable labeled 'dW'
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    # multipy the decay rate
    return tf.multiply(0.0002, tf.add_n(costs))

def conv(name,x,filter_size, in_filters, out_filters, strides):
    # Note: the input size should be:[batch, in_height, in_width, in_channels]
    # the filter size should be: [filter_height, filter_width, in_channels, out_channels]
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
      # random initialization
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],tf.float32,initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')


def batch_norm(name,x,is_training):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x,center=False,scale=False,training=is_training)

def global_avg_pool(x):
    pool1=tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=1)
    return pool1
