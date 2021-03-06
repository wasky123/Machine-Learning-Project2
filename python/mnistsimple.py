
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_dir='/tmp/tensorflow/mnist/input_data'
log_dir='/home/wasky/tensorflow-r1.4/tensorflow/examples/tutorials/mnist/tensor/mnist' 


def make_hparam_string(learning_rate):

    return "lr_%.0E" % (learning_rate)

for learning_rate in [1E-4]:
  hparam = make_hparam_string(learning_rate)
  print('Starting run for %s' % hparam)
  max_steps=3000
  #learning_rate=0.001
  dropout=0.8 


  mnist = input_data.read_data_sets(data_dir,one_hot=True)

  sess = tf.InteractiveSession() 

  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10) 


  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):

    with tf.name_scope(layer_name):

      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights) 
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases) 

      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate) 
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations) 
      return activations

  hidden1 = nn_layer(x, 784, 300, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob) 
    dropped = tf.nn.dropout(hidden1, keep_prob)



  y = nn_layer(dropped, 300, 10, 'layer2', act=tf.identity)
  y_predict=tf.nn.softmax(y)
  with tf.name_scope('cross_entropy'):

    diff = -tf.reduce_sum(y_ * tf.log(y_predict),reduction_indices=[1])
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff) 
  tf.summary.scalar('cross_entropy', cross_entropy) 

  with tf.name_scope('train'):
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy) 


  merged = tf.summary.merge_all()  

  train_writer = tf.summary.FileWriter(log_dir + '/train'+ str(hparam), sess.graph)
  test_writer = tf.summary.FileWriter(log_dir + '/test'+ str(hparam)) 
  tf.global_variables_initializer().run() 


  def feed_dict(train): 

    if train:
      xs, ys = mnist.train.next_batch(100)
      k = dropout 
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0 
    return {x: xs, y_: ys, keep_prob: k}


  saver = tf.train.Saver()  
  for i in range(max_steps):
    if i % 100 == 0:  
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      #train_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  
      if i % 100 == 99:  
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata() 
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)

        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        saver.save(sess, log_dir+"/model.ckpt", i)
        print('Adding run metadata for', i)
      else:  
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

#train_writer.close()
#test_writer.close()






print('Done training!')

