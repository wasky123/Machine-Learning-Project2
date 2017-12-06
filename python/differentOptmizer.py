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
    y_actualtrain = mnist.train.labels[:num,:]
    #print ('y_actualtrain Examples Loaded = ' + str(y_actualtrain.shape))

    return x_train, y_actualtrain

def Valid_SIZE(num):

    x_valid = mnist.train.images[50000:num,:]
    #print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_actualvalid = mnist.train.labels[50000:num,:]
    #print ('y_actualtrain Examples Loaded = ' + str(y_actualtrain.shape))

    return x_valid, y_actualvalid

def TEST_SIZE(num):
    x_test = mnist.test.images[:num,:]
    y_actualtest = mnist.test.labels[:num,:]

    return x_test, y_actualtest


x_train, y_train = TRAIN_SIZE(50000)
x_valid, y_valid = Valid_SIZE(55000)
x_test, y_test = TEST_SIZE(10000)
trainepoch = 3000
batch_size = 100
LEARNING_RATE = 0.3


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()

in_units=784
h1_units=300
h2_units =300

W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.random_normal([h1_units, h2_units]))
b2=tf.Variable(tf.zeros([h2_units]))
W3=tf.Variable(tf.zeros([h2_units,10]))
b3=tf.Variable(tf.zeros([10]))
#init_momentum = 0.9
global_step = tf.Variable(0, trainable=False)
#momentum = tf.Variable(init_momentum, trainable=False)

x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)

#sigma function
#hidden1=tf.nn.sigmoid(tf.matmul(x,W1) + b1)
#hidden2=tf.nn.sigmoid(tf.matmul(hidden1,W2) + b2)
#y_predict=tf.nn.softmax(tf.matmul(hidden2,W3)+b3)
#with dropout and relu
hidden1=tf.nn.relu(tf.matmul(x,W1) + b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
#y=tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
hidden2=tf.nn.relu(tf.matmul(hidden1_drop,W2) + b2)
hidden2_drop=tf.nn.dropout(hidden1,keep_prob)
#softmax classifier
y_predict=tf.nn.softmax(tf.matmul(hidden2_drop,W3)+b3)
y_actual=tf.placeholder(tf.float32,[None,10])

#loss function
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict),reduction_indices=[1]))
#mse = tf.reduce_mean(tf.square(y_predict - y_actual))

#optimizer
#train_step=tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(LEARNING_RATE,0.9).minimize(cross_entropy,global_step=global_step)                                              

correct_prediction=tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
a = []
c = []
d = []
e = []
f = []

#train process
for i in range(trainepoch):
    #sess.run(train_step, feed_dict={x: x_train, y_actual: y_train})
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    #train_step.run({x:batch_xs,y_actual:batch_ys,keep_prob:1.0})
    train_step.run({x:batch_xs,y_actual:batch_ys,keep_prob:0.75})


    if(i%50==0 or i==trainepoch-1):
        print ("cross_etpy:"+str(sess.run(cross_entropy, {x: x_train, y_actual: y_train,keep_prob:1.0}))+" valid:"+str(sess.run(cross_entropy, {x: x_valid, y_actual: y_valid,keep_prob:1.0}))+" valid_accuracy:"+str(accuracy.eval({x: x_valid, y_actual: y_valid,keep_prob:1.0}))+" test_accuracy:"+str(accuracy.eval({x: x_test, y_actual: y_test,keep_prob:1.0})))
        #training_cross_entropy= sess.run(cross_entropy, {x: x_train, y_actual: y_train,keep_prob:1.0})
        #valid_cross_entropy = sess.run(cross_entropy, {x: x_valid, y_actual: y_valid,keep_prob:1.0})
        test_accuracy = accuracy.eval({x: x_test, y_actual: y_test,keep_prob:1.0})
        train_accuracy = accuracy.eval({x: x_train, y_actual: y_train,keep_prob:1.0})
        valid_accuracy = accuracy.eval({x: x_valid, y_actual: y_valid,keep_prob:1.0})
        #a.append(training_cross_entropy)
        #c.append(valid_cross_entropy)
        e.append(train_accuracy)
        d.append(test_accuracy)
        f.append(valid_accuracy)

        #print ("trainepoch "+str(i)+" cross_entropy train:"+str(sess.run(cross_entropy, {x: x_train, y_actual: y_train,keep_prob:1.0})))
        #print ("train_accuracy:",accuracy.eval({x:mnist.train.images,y_actual:mnist.train.labels,keep_prob:1.0}))
        #print ("valid_accuracy:",accuracy.eval({x: x_valid, y_actual: y_valid,keep_prob:1.0}))
        #print ("test_accuracy:",accuracy.eval({x:mnist.test.images,y_actual:mnist.test.labels,keep_prob:1.0})) 
    #if(i%100==0):

        #print(accuracy.eval({x:mnist.test.images,y_actual:mnist.test.labels,keep_prob:1.0}))

'''acc = accuracy.eval({x:mnist.test.images,y_actual:mnist.test.labels,keep_prob:1.0})
b = []
for i in range(0,61):
    b.append(i*50)
plt.plot(b,a,'r',label="train_loss")
plt.plot(b,c,'g',label="valid_loss")

plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('cross_entropy')
plt.title('lr=%f, te=%d, bs=%d, acc=%f' % (LEARNING_RATE, trainepoch, batch_size, acc))
plt.tight_layout()
plt.show()'''

b = []
for i in range(0,61):
    b.append(i*50)
plt.plot(b,e,'b',label="train_accuracy")
plt.plot(b,d,'r',label="test_accuracy")
plt.plot(b,f,'g',label="valid_accuracy")

plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.tight_layout()
plt.show()  













