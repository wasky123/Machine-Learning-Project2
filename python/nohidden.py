import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import math
import numpy as np
import random as ran
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def TRAIN_SIZE(num):

    x_train = mnist.train.images[:num,:]
    #print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    #print ('y_train Examples Loaded = ' + str(y_train.shape))

    return x_train, y_train

def Valid_SIZE(num):

    x_valid = mnist.train.images[50000:num,:]
    #print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_valid = mnist.train.labels[50000:num,:]
    #print ('y_train Examples Loaded = ' + str(y_train.shape))

    return x_valid, y_valid

def TEST_SIZE(num):
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]

    return x_test, y_test


x_train, y_train = TRAIN_SIZE(50000)
x_valid, y_valid = Valid_SIZE(55000)
x_test, y_test = TEST_SIZE(10000)
trainstep = 10000
LEARNING_RATE = 0.1
x = tf.placeholder(tf.float32, [None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))        
b = tf.Variable(tf.zeros([10]))            
y_predict = tf.nn.softmax(tf.matmul(x,W) + b)     

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual* tf.log(y_predict), reduction_indices=[1]))
  
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)   

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))   
             
init = tf.initialize_all_variables()
#file = open('nohiddenfinalW.txt', 'a')

def as_num(x):
    y='{:.5f}'.format(x)
    return(y)

with tf.Session() as sess:
    sess.run(init)
    #x_train, y_train = TRAIN_SIZE(3)
     
    for i in range(trainstep):              
        batch_xs, batch_ys = mnist.train.next_batch(100)          
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})
        #y_trainpredict = sess.run(y_predict, feed_dict={x: x_train})
        #y_validpredict = sess.run(y_predict, feed_dict={x: x_valid})
        if(i%100==0):
            print ("trainStep "+str(i)+" cross_entropy train:"+str(sess.run(cross_entropy, {x: x_train, y_actual: y_train}))+" valid:"+str(sess.run(cross_entropy, {x: x_valid, y_actual: y_valid})))

            print ("train_accuracy:",sess.run(accuracy, feed_dict={x: mnist.train.images, y_actual: mnist.train.labels}))
            print ("test_accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))   
        #if(i%100==0):

            #print ("train_accuracy:",sess.run(accuracy, feed_dict={x: mnist.train.images, y_actual: mnist.train.labels}))
            #print ("cross_entropy:",sess.run(cross_entropy, {x: x_train, y_actual: y_train}))
            #print ("valid_accuracy:",sess.run(accuracy, feed_dict={x: x_valid, y_actual: y_valid}))                  
            #print ("test_accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))
            #print(sess.run(y_predict, feed_dict={x: x_train}))
        '''elif i==999:
            finalW = sess.run(W)
            np.savetxt('nohiddenfinalb', sess.run(b), delimiter=",")
            W1 = np.zeros((784,10))
            for j in range(784):
                for k in range(10):
                    W1[j,k] = as_num(finalW[j,k])
                    file.write(str(W1[j,k])+',')
                file.write('\n')'''
