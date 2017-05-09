from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_optimizer(name):
    if 'sgd' in name.lower():
        optimizer = tf.train.GradientDescentOptimizer
    elif 'momentum' in name.lower():
        optimizer = tf.train.MomentumOptimizer
    elif 'adagrad' in name.lower():
        optimizer = tf.train.AdagradOptimizer
    elif 'adadelta' in name.lower():
        optimizer = tf.train.AdadeltaOptimizer
    elif 'adam' in name.lower():
        optimizer = tf.train.AdamOptimizer
    else:
        raise ValueError('unsupported optimizer')
    return optimizer

def get_act_function(name):
    if 'relu' in name.lower():
        act_function = tf.nn.relu
    elif 'softplus' in name.lower():
        act_function = tf.nn.softplus
    elif 'sigmoid' in name.lower():
        act_function = tf.nn.sigmoid
    elif 'tanh' in name.lower():
        act_function = tf.nn.tanh
    else:
        raise ValueError('unsupported activation')
    return act_function

# initial variable
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Multilayer Perceptron
def nn_layer(input_tensor, input_dim, output_dim,
             act_function=tf.nn.relu, layer_name='Layer'):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            # variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            # variable_summaries(biases)
        with tf.name_scope('WX_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # tf.summary.histogram('pre_activations', preactivate)
        activations = act_function(preactivate, name='activations')
        tf.summary.histogram('activations', activations)
    return activations, weights, biases

# summary variables
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# ----------------------------- #
def mlp_run(log_dir, max_epoch=10000, mnist=None,
            dropout=True, dropout_rate=0.5,
            learning_rate=0.001, decay_lr=False,
            act_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer):

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # read mnist
    if not mnist:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # initial session
    sess = tf.Session()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1, W1, _ = nn_layer(x, input_dim=784, output_dim=500,
                              act_function=act_function,
                              layer_name='Layer_1')

    if dropout:
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden1, keep_prob)
    else:
        keep_prob = tf.placeholder(tf.float32)
        dropped = hidden1
        dropout_rate = 1.0

    # Readput layer
    y, W_out, _ = nn_layer(dropped, input_dim=500, output_dim=10,
                           act_function=tf.identity,
                           layer_name='Readout_Layer')

    # loss function
    with tf.name_scope('Loss_function'):
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
            loss_function = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', loss_function)

    # train and evaluate model
    if decay_lr:
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.01
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=100, decay_rate=0.96,
                                                   staircase=True)
        with tf.name_scope('train'):
            train_step = optimizer(learning_rate).minimize(loss_function, global_step=global_step)
    else:
        with tf.name_scope('train'):
            train_step = optimizer(learning_rate).minimize(loss_function)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # merge tf.summary operation
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

    tf.global_variables_initializer().run()

    # define feed_dict loss function
    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = dropout_rate
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # running and logging
    for i in xrange(max_epoch):
        if i % 100 == 0:
            train_summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(train_summary, i)
            test_summary, test_acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(test_summary, i)
            # print('Testing Accuracy at step {0}: {1}'.format(i, test_acc))
        else:
            train_summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(train_summary, i)

    test_summary, test_acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(test_summary, i)
    train_writer.close()
    test_writer.close()
    # reset graph
    tf.reset_default_graph()
    return test_acc

if __name__ == '__main__':

    # read dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # training
    log_dir = '/tmp/tensorflow/mnist/logs/MLP'
    acc = mlp_run(log_dir, mnist=mnist)
    print('test accuracy for MLP :', acc)
