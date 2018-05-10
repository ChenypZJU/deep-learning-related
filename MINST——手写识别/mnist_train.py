# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:07:27 2018

@author: chenyiping
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  #示例，不适用与其他数据集

import mnist_inference


BATCH_SIZE=100

learning_rate_base=0.8
learning_rate_decay=0.99

regularization_rate=0.0001
training_steps=20000
moving_average_decay=0.99


def train(mnist):
    #搭建三层神经网络结构
    x=tf.placeholder(tf.float32,[None,mnist_inference.input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,mnist_inference.output_node],name='y-input')
    
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    
    y=mnist_inference.inference(x,regularizer)
    
    global_step=tf.Variable(0,trainable=False)
    
    #对可训练参数使用滑动平均，这里是variable型的 weight1，biases1，weight2，biases2
    variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    #滑动平均模型的输出
    #average_y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    #L2正则化，加上平均交叉熵为损失函数值
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses')) #与minst_inference相对应
    
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/BATCH_SIZE,learning_rate_decay)
    
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    #这里讲一次更新完成更新参数train_step和更新每一个参数的滑动平均值variable_averages_op两个操作
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        #validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        #test_feed={x:mnist.test.images,y_:mnist.test.labels}
        
        for i in range(training_steps):
            #每次迭代训练一个batch
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
            
            if i%1000==0:
                #validation_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),loss on training batch is %g"%(step,loss_value))
                saver.save(sess,"Model/model.ckpt",global_step)
            
        
def main(argv=None):
    mnist=input_data.read_data_sets("D:\机器学习\深度学习\MINST——手写识别",one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    tf.app.run()
        