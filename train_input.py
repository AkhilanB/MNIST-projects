from keras.datasets import mnist
from sklearn.utils import resample
import numpy as np
import tensorflow as tf
import sys
from helper.networks import conv_net, optimize
from helper.utils import display

RESTORE = sys.argv[1] == 'True' #All other args are treated as False
TOTAL_ITERS = 10
#For each overall iteration, train model until returns False
MODEL_TRAIN_CONDITION = lambda it, acc: acc < .99 and it < 300
#For each overall iteration, train input until returns False
INPUT_TRAIN_CONDITION = lambda it, acc: acc < .95 and it < 200
BATCH_SIZE = 300
#Number of input trained batches added to dataset
#600 recommended for class balance
NUM_VAR_INPUTS = 600

#Get data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Scale images
x_train, x_test = x_train/255, x_test/255
#Convert to one hot encoding
#There are 11 categories: 10 digits + 1 for input trained images
y_train = np.eye(11)[y_train]
y_test = np.eye(11)[y_test]



#Label for MNIST digit
labels = tf.placeholder('float', shape=(None, 11))

#Inputs fed for training model
fed_inputs = tf.placeholder('float', shape=[None, 28, 28])

#Variable to train when training input
with tf.variable_scope('Input') as scope:
    var_inputs = tf.Variable(tf.zeros([11, 28, 28]))
scaled_var_inputs = tf.nn.sigmoid(var_inputs)

with tf.variable_scope('Model') as scope:
    model_logits = conv_net(fed_inputs) #Used when training model
    scope.reuse_variables()
    input_logits = conv_net(scaled_var_inputs) #Used when training input

#Train conv_net to match inputs to labels
var_list = [v for v in tf.trainable_variables() if v.name.startswith('Model/')]
model_optimizer, model_accuracy, model_cross_entropy = optimize(model_logits, labels,
                                                                var_list, learning_rate = 0.01)

#Train inputs to match logits to labels
var_list = [v for v in tf.trainable_variables() if v.name.startswith('Input/')]
input_optimizer, input_accuracy, input_cross_entropy = optimize(input_logits, labels,
                                                                var_list, learning_rate = 0.1)



#Labels to train input
input_labels = np.identity(11) #Labels are each of the digits + label for input trained images
#Labels for input trained images
error_labels = np.zeros((10,11))
error_labels[:,10] = 1

saver = tf.train.Saver()
with tf.Session() as sess:
    if RESTORE:
        saver.restore(sess, './train_input.ckpt')
    else:
        sess.run(tf.global_variables_initializer())

    trained_data = None
    
    for iteration in range(TOTAL_ITERS):
        accuracy_value = 0
        sub_iter = 0
        while MODEL_TRAIN_CONDITION(sub_iter, accuracy_value):
            if trained_data is None: #Add trained_data if available
                batch_data, batch_labels = resample(x_train, y_train, n_samples = BATCH_SIZE)
            else:
                full_x_train = np.vstack((x_train, trained_data))
                full_y_train = np.vstack((y_train, np.vstack([error_labels]*NUM_VAR_INPUTS)))
                batch_data, batch_labels = resample(x_train, y_train, n_samples = BATCH_SIZE)
            
            _, cross_entropy_value, accuracy_value = sess.run([model_optimizer, model_cross_entropy, model_accuracy],
                                                              feed_dict={labels: batch_labels, fed_inputs: batch_data})
            print('/'.join([str(iteration),str(TOTAL_ITERS)]) + ' Model Train Loss, Accuracy : '
                  + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')
            #End is carraige return to overwrite previous output
            sub_iter += 1

        accuracy_value = 0
        sub_iter = 0
        while INPUT_TRAIN_CONDITION(sub_iter, accuracy_value):
            _, cross_entropy_value, accuracy_value = sess.run([input_optimizer, input_cross_entropy, input_accuracy],
                                                              feed_dict={labels: input_labels})
            print('/'.join([str(iteration),str(TOTAL_ITERS)]) + ' Input Train Loss, Accuracy : '
                  + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')
            #End is carraige return to overwrite previous output
            sub_iter += 1

        var_inputs_value  = sess.run(scaled_var_inputs)[:10,:,:] #Retaining only the images corresponding to digits
        
        if trained_data is None: #If there is no trained_data, sets it to values from var_input_values
            trained_data = np.vstack([var_inputs_value]*NUM_VAR_INPUTS)
        else: #Combines old trained_data with var_input_value based on iteration number
            num_new_inputs = int(NUM_VAR_INPUTS/(iteration+1))
            num_old_inputs = NUM_VAR_INPUTS-num_new_inputs
            new_trained_data = np.vstack([var_inputs_value]*num_new_inputs)
            old_trained_data = resample(trained_data, n_samples = 10*num_old_inputs)
            trained_data = np.vstack((old_trained_data, new_trained_data))
        

    cross_entropy_value, accuracy_value = sess.run([model_cross_entropy, model_accuracy],
                                                  feed_dict={labels: y_test, fed_inputs: x_test})
    
    print('') #Newline so that the previous output is not overwritten
    
    print(' Model Test Loss, Accuracy : ' + ' '.join([str(cross_entropy_value),str(accuracy_value)]), end = '\r')

    var_inputs_value  = sess.run(scaled_var_inputs)
    
    display([var_inputs_value[i,:,:] for i in range(11)])

    saver.save(sess, './train_input.ckpt')

