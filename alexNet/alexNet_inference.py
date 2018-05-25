# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:48:45 2018

@author: chenyiping

AlexNet 神经网络结构
"""

import tensorflow as tf

num_channels=3

conv1_deep=96
conv1_size=11
conv1_stride=4;

pool1_size=3
pool1_stride=2

conv2_deep=256
conv2_size=5
conv2_stride=1

pool2_size=3
pool2_stride=2

conv3_deep=384
conv3_size=3
conv3_stride=1

conv4_deep=384
conv4_size=3
conv4_stride=1

conv5_deep=256
conv5_size=3
conv5_stride=1

pool3_size=3
pool3_stride=2

fc1_size=4096
fc2_size=4096
fc3_size=1000
def inference(input_tensor,train,regularizer):
    '''
    该函数定义了AlexNet的神经网络结构
    参考论文：1)ImageNet Classification with Deep Convolutional Neural Networks,
              Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, 
              COMMUNICATIONS OF THE ACM,2017,6(60):84-90
             2) cs231n lecture 9
              https://www.youtube.com/watch?v=DAOcjicFr1Y&index=8&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
    这里输入张量维度为[batch_size,image_size,image_size,num_channels]
    '''
    with tf.variable_scope('layer1-conv1'):
        '''
        过滤器参数[filter_height, filter_width, in_channels, out_channels]
        在conv2d函数中，strides列表与输入形式相关，如果输入是“NHWC”， [batch, height, width, channels]
        那么通常strides=[1,stride,stride,1]
        #第一层卷积层，一共有96个过滤器，尺寸为11*11，不使用全0填充
        '''
        conv1_weights=tf.get_variable("weight",[conv1_size,conv1_size,num_channels,conv1_deep],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[conv1_deep],initializer=tf.constant_initilizer(0.0))
        
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,conv1_stride,conv1_stride,1],padding='VALID')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,pool1_size,pool1_size,1],strides=[1,pool1_stride,pool1_stride,1],padding='VALID')
    
    
    with tf.name_scope('layer3-norm1'):
        norm1=tf.nn.local_response_normalization(pool1,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)
        
    with tf.variable_scope('layer4-conv2'):
        
        conv2_weights=tf.get_variable("weight",[conv2_size,conv2_size,conv1_deep,conv2_deep],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("bias",[conv2_deep],initializer=tf.constant_initilizer(0.0))
        
        conv2=tf.nn.conv2d(norm1,conv2_weights,strides=[1,conv2_stride,conv2_stride,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    
    with tf.name_scope('layer5-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,pool2_size,pool2_size,1],strides=[1,pool2_stride,pool2_stride,1],padding='VALID')
        
    with tf.name_scope('layer6-norm2'):
        norm2=tf.nn.local_response_normalization(pool2,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)
    
    with tf.variable_scope('layer7-conv3'):
        conv3_weights=tf.get_variable("weight",[conv3_size,conv3_size,conv2_deep,conv3_deep],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases=tf.get_variable("bias",[conv3_deep],initializer=tf.constant_initilizer(0.0))
        
        conv3=tf.nn.conv2d(norm2,conv3_weights,strides=[1,conv3_stride,conv3_stride,1],padding='SAME')
        relu3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
    
    with tf.variable_scope('layer8-conv4'):
        conv4_weights=tf.get_variable("weight",[conv4_size,conv4_size,conv3_deep,conv4_deep],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases=tf.get_variable("bias",[conv4_deep],initializer=tf.constant_initilizer(0.0))
        
        conv4=tf.nn.conv2d(relu3,conv4_weights,strides=[1,conv4_stride,conv4_stride,1],padding='SAME')
        relu4=tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
        
    with tf.variable_scope('layer9-conv5'):
        conv5_weights=tf.get_variable("weight",[conv5_size,conv5_size,conv4_deep,conv5_deep],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases=tf.get_variable("bias",[conv5_deep],initializer=tf.constant_initilizer(0.0))
        
        conv5=tf.nn.conv2d(relu4,conv5_weights,strides=[1,conv5_stride,conv5_stride,1],padding='SAME')
        relu5=tf.nn.relu(tf.nn.bias_add(conv5,conv5_biases))
        
    with tf.name_scope('layer10-pool3'):
        pool3=tf.nn.max_pool(relu5,ksize=[1,pool3_size,pool3_size,1],strides=[1,pool3_stride,pool3_stride,1],padding='VALID')     
    
    #flattern
    pool3_shape=pool3.get_shape().as_list()
    nodes=pool3_shape[1]*pool3_shape[2]*pool3_shape[3]    
    reshaped=tf.reshape(pool3,[pool3_shape[0],nodes])
    
    #全连接层，前两层有dropout
    with tf.variable_scope('layer11-fc1'):
        fc1_weights=tf.get_variable("weight",[nodes,fc1_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        biases=tf.get_variable('bias',[fc1_size],initializer=tf.constant_initilizer(0.0))
        
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
            
    with tf.variable_scope('layer12-fc2'):
        fc2_weights=tf.get_variable("weight",[fc1_size,fc2_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        biases=tf.get_variable('bias',[fc2_size],initializer=tf.constant_initilizer(0.0))
        
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weights)+biases)
        if train:
            fc2=tf.nn.dropout(fc2,0.5)
    
    with tf.variable_scope('layer13-fc3'):
        fc3_weights=tf.get_variable("weight",[fc2_size,fc3_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        biases=tf.get_variable('bias',[fc3_size],initializer=tf.constant_initilizer(0.0))
        
        fc3=tf.matmul(fc2,fc3_weights)+biases  
        #输出之后与softmax和交叉熵cross entropy函数计算损失函数值
        #tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        
    return fc3
    
