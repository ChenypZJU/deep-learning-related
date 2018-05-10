# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:54:40 2018

@author: chenyiping
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_train
import mnist_inference

eval_interval_secs=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_inference.input_node],name='x-input')
        y_=tf.placeholder(tf.float32,[None,mnist_inference.output_node],name='y-input')
        
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        
        y=mnist_inference.inference(x,None)
        
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        variable_averages=tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        #生成variable_averages的应用变量（具体看mnist_train里的variable_averages.apply()函数）重命名
        variable_to_restore=variable_averages.variables_to_restore()  
        #print(variable_to_restore)
        saver=tf.train.Saver(variable_to_restore) #只有模型的平均滑动变量被加载进来                
        with tf.Session() as sess:            
            ckpt=tf.train.get_checkpoint_state('Model/')
            #print(ckpt)
            if ckpt and ckpt.all_model_checkpoint_paths:
                #加载模型
                #这一部分是有多个模型文件时，对所有模型进行测试验证
                for path in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess,path)                
                    global_step=path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training step(s),valisation accuracy = %g"%(global_step,accuracy_score))
                '''
                #对最新的模型进行测试验证
                saver.restore(sess,ckpt.model_checkpoint_paths)                
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                print("After %s training step(s),valisation accuracy = %g"%(global_step,accuracy_score))
                '''
            else:
                print('No checkpoint file found')
                return
        #time.sleep(eval_interval_secs)
        return
            
def main(argv=None):
    mnist=input_data.read_data_sets("D:\机器学习\深度学习\MINST——手写识别",one_hot=True)
    evaluate(mnist)
    
if __name__=='__main__':
    tf.app.run()
        
