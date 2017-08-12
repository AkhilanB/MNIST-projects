from keras.datasets import mnist
from sklearn.utils import resample
import numpy as np
import tensorflow as tf
import sys
import os
from helper.networks import conv_net, optimize
from helper.utils import display

RESTORE = sys.argv[1] == 'True' #All other args are treated as False
TOTAL_ITERS = 20
#For each overall iteration, train parity vars until returns False
PARITY_TRAIN_CONDITION = lambda it, acc: acc < .999 and it < 300
#For each overall iteration, train loops vars until returns False
LOOPS_TRAIN_CONDITION = lambda it, acc: acc < .95 and it < 300
#Weight given to parity cross entropy when training loops vars
PARITY_WEIGHT = 0.01
BATCH_SIZE = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Scale images
x_train, x_test = x_train/255, x_test/255
#Convert to one hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


inputs = tf.placeholder('float', shape=(None, 28, 28))

labels = tf.placeholder('float', shape=(None, 10))

loops = tf.constant(np.array([[0,1],[1,0],[1,0],
                              [1,0],[0,1],[1,0],
                              [0,1],[1,0],[0,1],
                              [0,1]]), dtype = 'float')

odd = tf.constant(np.array([[1,0],[0,1],[1,0],
                            [0,1],[1,0],[0,1],
                            [1,0],[0,1],[1,0],
                            [0,1]]), dtype = 'float')

#Label for whether digit has loop
labels_loops = labels @ loops
#Label for digit parity
labels_parity = labels @ odd

with tf.variable_scope('Loops') as scope:
    encoded = conv_net(inputs, n_logits = 100)
    logits_loops = tf.layers.dense(encoded, units = 2)

with tf.variable_scope('Parity') as scope:
    logits_parity = tf.layers.dense(encoded, units = 2)

#Train dense layer to match inputs to labels_parity
var_list = [v for v in tf.trainable_variables() if v.name.startswith('Parity/')]
parity_optimizer, parity_accuracy, parity_cross_entropy = optimize(logits_parity, labels_parity,
                                                                   var_list, learning_rate = 0.01)

#Train conv_net + dense layer to match inputs to labels_loops AND increase parity cross_entropy
var_list = [v for v in tf.trainable_variables() if v.name.startswith('Loops/')]
loops_optimizer, loops_accuracy, loops_cross_entropy = optimize(logits_loops, labels_loops,
                                                                var_list, learning_rate = 0.01,
                                                                additional_var = PARITY_WEIGHT*tf.negative(parity_cross_entropy))


saver = tf.train.Saver()
with tf.Session() as sess:
    if RESTORE:
        saver.restore(sess, './unbiased_digits.ckpt')
    else:
        sess.run(tf.global_variables_initializer())

    for iteration in range(TOTAL_ITERS):    
        accuracy_value = 0
        sub_iter = 0
        while LOOPS_TRAIN_CONDITION(sub_iter, accuracy_value):
            batch_data, batch_labels = resample(x_train, y_train, n_samples = BATCH_SIZE)
            feed_dict_train = {inputs: batch_data, labels: batch_labels}
            _, cross_entropy_value, accuracy_value = sess.run([loops_optimizer, loops_cross_entropy,
                                                               loops_accuracy], feed_dict=feed_dict_train)
            print('/'.join([str(iteration),str(TOTAL_ITERS)]) + ' Loops Train Loss, Accuracy : '
                  + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')
            #End is carraige return to overwrite previous output
            sub_iter += 1
        
        accuracy_value = 0
        sub_iter = 0
        while PARITY_TRAIN_CONDITION(sub_iter, accuracy_value):
            batch_data, batch_labels = resample(x_train, y_train, n_samples = BATCH_SIZE)
            feed_dict_train = {inputs: batch_data, labels: batch_labels}
            labels_val, _, cross_entropy_value, accuracy_value = sess.run([labels_loops, parity_optimizer, parity_accuracy,
                                                               parity_cross_entropy], feed_dict=feed_dict_train)
            print('/'.join([str(iteration),str(TOTAL_ITERS)]) + ' Parity Train Loss, Accuracy : '
                  + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')
            #End is carraige return to overwrite previous output
            sub_iter += 1
    
    cross_entropy_value, accuracy_value = sess.run([parity_cross_entropy, parity_accuracy],
                                                  feed_dict={labels: y_test, inputs: x_test})
    print('') #Newline so that the previous output is not overwritten
    print(' Parity Test Loss, Accuracy : ' + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')

    cross_entropy_value, accuracy_value = sess.run([loops_cross_entropy, loops_accuracy],
                                                  feed_dict={labels: y_test, inputs: x_test})
    print('') #Newline so that the previous output is not overwritten
    print(' Loops Test Loss, Accuracy : ' + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')

    saver.save(sess, './unbiased_digits.ckpt')

