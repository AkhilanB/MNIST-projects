from keras.datasets import mnist
from sklearn.utils import resample
import numpy as np
import tensorflow as tf
import sys
import itertools
from helper.networks import gen_conv_net, conv_net, linear_conv_net, optimize
from helper.utils import display

RESTORE = sys.argv[1] == 'True' #All other args are treated as False
TOTAL_ITERS = 10
#For each overall iteration, train encoder until returns False
ENCODER_TRAIN_CONDITION = lambda it: it < 1
#For each overall iteration, train decoder until returns False
DECODER_TRAIN_CONDITION = lambda it: it < 20
BATCH_SIZE = 1000

#Get data
(x_train, _), (x_test, _) = mnist.load_data()
#Scale images
x_train, x_test = x_train/255, x_test/255


#Inputs fed for training model
inputs = tf.placeholder('float', shape=[None, 28, 28])
inputs_flat = tf.reshape(inputs, (-1, 28 * 28))

with tf.variable_scope('Encode') as scope:
    encoded = linear_conv_net(inputs, n_logits = 50)

with tf.variable_scope('Decode') as scope:
    decoded = gen_conv_net(encoded)

with tf.variable_scope('Encode', reuse = True) as scope:
    reencoded = linear_conv_net(decoded, n_logits = 50)

decoded_flat = tf.reshape(decoded, (-1, 28 * 28))

image_match = tf.nn.sigmoid_cross_entropy_with_logits(logits = decoded_flat, labels = inputs_flat)
encodings_match = tf.losses.mean_squared_error(labels = encoded, predictions = reencoded)

#Train encoder to match inputs to decoded
var_list = [v for v in tf.trainable_variables() if v.name.startswith('Encode/')]
encoder_loss = tf.reduce_mean(encodings_match)
encoder_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(encoder_loss, var_list=var_list)

#Train decoder to match inputs to decoded
var_list = [v for v in tf.trainable_variables() if v.name.startswith('Decode/')]
decoder_loss = tf.reduce_mean(image_match)
decoder_optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(decoder_loss, var_list=var_list)

saver = tf.train.Saver()
with tf.Session() as sess:
    if RESTORE:
        saver.restore(sess, './autoencode.ckpt')
    else:
        sess.run(tf.global_variables_initializer())

    for iteration in range(TOTAL_ITERS):   
        sub_iter = 0
        while ENCODER_TRAIN_CONDITION(sub_iter):
            batch_data = resample(x_train, n_samples = BATCH_SIZE)
            
            _, loss_value,  = sess.run([encoder_optimizer, encoder_loss],
                                              feed_dict={inputs: batch_data})
            print('/'.join([str(iteration),str(TOTAL_ITERS)]) + ' Encoder Loss: '
                  + str(loss_value), end = '\r')
            #End is carraige return to overwrite previous output
            sub_iter += 1

        sub_iter = 0
        while DECODER_TRAIN_CONDITION(sub_iter):
            batch_data = resample(x_train, n_samples = BATCH_SIZE)
            
            _, loss_value = sess.run([decoder_optimizer, decoder_loss],
                                              feed_dict={inputs: batch_data})
            print('/'.join([str(iteration),str(TOTAL_ITERS)]) + ' Decoder Loss: '
                  + str(loss_value), end = '\r')
            #End is carraige return to overwrite previous output
            sub_iter += 1
            
    display_data = resample(x_test, n_samples = 9)
            
    decoded_data = sess.run(decoded, feed_dict={inputs: display_data})
    display([display_data[i,:,:] for i in range(9)])
    display([decoded_data[i,:,:] for i in range(9)])

    saver.save(sess, './autoencode.ckpt')

