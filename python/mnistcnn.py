from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os
import sys

import tensorflow as tf
FLAGS = None

LOGDIR = "/tmp/mnist_tutorial/"
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

def conv_layer(input, 
               channels_in,    
               channels_out,   
               name='conv'):   
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], 
                                                  stddev=0.05), name='W')
        biases = tf.Variable(tf.constant(0.05, shape=[channels_out]), name='B')
        conv = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + biases)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activations', act)

        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc_layer(input, num_inputs, num_outputs, name='fc'):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], 
                                                  stddev=0.05), name='W')
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name='B')
        act = tf.matmul(input, weights) + biases

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activations', act)

        return act

LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")

tensorboard_dir = 'tensorbo234/mnist'   
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


def mnist_model(learning_rate, use_two_fc, use_two_conv, hparam):
    tf.reset_default_graph()    
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    if use_two_conv:   
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")    
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if use_two_fc:    
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        relu = tf.nn.relu(fc1)
        embedding_input = relu
        tf.summary.histogram("fc1/relu", relu)
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
        embedding_input = flattened  
        embedding_size = 7*7*64
        logits = fc_layer(flattened, 7*7*64, 10, "fc")

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="cross_entropy")
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()    

    
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()     

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(tensorboard_dir + hparam)
    test_writer = tf.summary.FileWriter(tensorboard_dir + '/test'+ str(hparam))
    writer.add_graph(sess.graph)

  
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:   
            
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)

        if i % 100 == 0:
            summary, acc = sess.run([summ, accuracy], feed_dict={x:mnist.test.images, y:mnist.test.labels})
            test_writer.add_summary(summary, i) 

            print('Test Accuracy at step %s: %s' % (i, acc))   
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            saver.save(sess, os.path.join(tensorboard_dir, "model.ckpt"), i)

            print("iteration: {0:>6}, accuracy: {1:>6.4%}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})



def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

for learning_rate in [1e-4]:
    for use_two_fc in [True]:
        for use_two_conv in [True]:
            hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
            print('Starting run for %s' % hparam)

            mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)

print('Done training!')



