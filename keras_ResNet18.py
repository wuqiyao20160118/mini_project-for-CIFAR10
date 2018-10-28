import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os
from time import time
from datetime import datetime

def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation='relu'):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
           kernel_initializer='TruncatedNormal',kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation('relu')(x)
    return x

# 建一个20层的ResNet网络 
def resnet_v1(input_shape):
    #input_shape should be (width,height,channel)
    inputs = Input(shape=input_shape)# Input层，用来当做占位使用

    #第一层
    x = resnet_block(inputs)
    print('layer1,xshape:',x.shape)
    # 第2~7层
    for i in range(6):
        a = resnet_block(inputs = x)
        b = resnet_block(inputs=a,activation=None)
        x = keras.layers.add([x,b])
        x = Activation('relu')(x)
    # out：32*32*16
    # 第8~13层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs = x,strides=2,num_filters=32)
        else:
            a = resnet_block(inputs = x,num_filters=32)
        b = resnet_block(inputs=a,activation=None,num_filters=32)
        if i==0:
            x = Conv2D(32,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='TruncatedNormal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])
        x = Activation('relu')(x)
    # out:16*16*32
    # 第14~19层
    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x,strides=2,num_filters=64)
        else:
            a = resnet_block(inputs = x,num_filters=64)

        b = resnet_block(inputs=a,activation=None,num_filters=64)
        if i == 0:
            x = Conv2D(64,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='TruncatedNormal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])# 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
    # out:8*8*64
    # 第20层   
    x = AveragePooling2D(pool_size=2,strides=1)(x)
    # out:7*7*64
    y = Flatten()(x)
    # out:1024
    outputs = Dense(10,activation='softmax',
                    kernel_initializer='TruncatedNormal')(y)

    #初始化模型
    #之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = Model(inputs=inputs,outputs=outputs)
    return model

def train(batch, epoch, X_train, y_train, X_test, y_test):
    model = resnet_v1((32,32,3))
    #对学习过程进行配置
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    model.summary()
    start = time()
    #make log_dir
    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    
    #configuring callbacks
    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,mode='max',min_lr=1e-3)

    mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5', 
                         monitor='val_acc', save_best_only=True)
    #use the TensorBoard
    tb = TensorBoard(log_dir=log_dir, )
    
    #implementing image augmentation
    aug = ImageDataGenerator(width_shift_range = 0.125, height_shift_range = 0.125, horizontal_flip = True)
    aug.fit(X_train)
    gen = aug.flow(X_train, y_train, batch_size=batch)
    
    #build the model by using fit_generator()
    h = model.fit_generator(generator=gen, steps_per_epoch=50000//batch, epochs=epoch, validation_data=(X_test, y_test),callbacks=[mc, tb,lr_scheduler,lr_reducer]) #save the storage!
    print('\n@ Total Time Spent: %.2f seconds' % (time() - start))
    acc, val_acc = h.history['acc'], h.history['val_acc']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    return h

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']

    epoch = len(acc)

    plt.figure(figsize=(17, 5))

    plt.subplot(121)

    plt.plot(range(epoch), acc, label='Train')

    plt.plot(range(epoch), val_acc, label='Test')

    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)

    plt.legend()

    plt.grid(True)

    plt.subplot(122)

    plt.plot(range(epoch), loss, label='Train')

    plt.plot(range(epoch), val_loss, label='Test')

    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)

    plt.legend()

    plt.grid(True)

    plt.show()

def lr_sch(epoch,total_epoch=200):
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5
