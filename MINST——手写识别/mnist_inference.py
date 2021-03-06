# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:52:11 2018

@author: chenyiping
"""
import tensorflow as tf

input_node=784
output_node=10
layer1_node=500

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    #将weights加入losses集合
    if(regularizer !=None):
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights=get_weight_variable([input_node,layer1_node],regularizer)
        biases=tf.get_variable("biases",[layer1_node],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([layer1_node,output_node],regularizer)
        biases=tf.get_variable("biases",[output_node],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
    
    return layer2