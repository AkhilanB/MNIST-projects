import tensorflow as tf

#Creates a feedforwarrd neural network for encoding
def dense_net(inputs, n_logits = 11):
    input_layer = tf.reshape(inputs, (-1, 28 * 28))
    
    dense1 = tf.layers.dense(
        inputs = input_layer, units = 300,
        activation = tf.nn.softmax,
        name = 'dense1')

    dense2 = tf.layers.dense(
        inputs = dense1, units = 50,
        activation = tf.nn.softmax,
        name = 'dense2')
    
    logits = tf.layers.dense(
        inputs = dense2, units = n_logits,
        name = 'dense')
    return logits

#Creates a convolutional neural network with no activation function
def linear_conv_net(inputs, n_logits = 11):
    input_layer = tf.reshape(inputs, (-1, 28, 28, 1))
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32,
        kernel_size=(3, 3),padding='same',
        name = 'conv1')
    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=(2, 2),
        strides=(2,2), padding = 'same')
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,filters=32,
        kernel_size=(3, 3),padding='same',
        name = 'conv2')
    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=(2, 2),
        strides=(2,2), padding = 'same')
    
    conv3 = tf.layers.conv2d(
        inputs=pool2,filters=16,kernel_size=(3, 3),
        padding='same', name = 'conv3')
    
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3, pool_size=(2, 2),
        strides=(2,2), padding = 'same')
    
    pool3_flat = tf.reshape(pool3, (-1, 4 * 4 * 16))
    
    encoded =  tf.layers.dense(
        inputs = pool3_flat, units = n_logits, name = 'dense')

    return encoded

#Creates a convolutional neural network for encoding
def conv_net(inputs, n_logits = 11):
    input_layer = tf.reshape(inputs, (-1, 28, 28, 1))
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=64,
        kernel_size=(5, 5),padding='same',
        activation = tf.nn.softmax,
        name = 'conv1')
    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=(2, 2),
        strides=(2,2), padding = 'same')
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,filters=32,
        kernel_size=(4, 4),padding='same',
        activation = tf.nn.softmax,
        name = 'conv2')
    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=(2, 2),
        strides=(2,2), padding = 'same')
    
    pool2_flat = tf.reshape(pool2, (-1, 7 * 7 * 32))
    
    logits = tf.layers.dense(
        inputs = pool2_flat, units = n_logits,
        name = 'dense')
    return logits

#Creates a convolutional neural network for image generation
def gen_conv_net(inputs):
    linear = tf.layers.dense(
        inputs = inputs, units = 256,
        name = 'gen_dense')
    
    reshaped_linear = tf.reshape(linear, (-1, 4, 4, 16))
    
    upsample1 = tf.image.resize_images(
        reshaped_linear, size = (7, 7),
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    conv1 = tf.layers.conv2d(
        inputs=upsample1, filters=32,
        kernel_size=(3,3), padding='same',
        name = 'gen_conv1')
    
    upsample2 = tf.image.resize_images(
        conv1, size = (14, 14),
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    conv2 = tf.layers.conv2d(
        inputs=upsample2, filters=16,
        kernel_size=(3,3), padding='same',
        name = 'gen_conv2')
    
    upsample3 = tf.image.resize_images(
        conv2, size = (28, 28),
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    conv3 = tf.layers.conv2d(
        inputs=upsample3, filters=1,
        kernel_size=(3,3), padding='same',
        name = 'gen_conv3')
    
    image = tf.reshape(conv3, (-1, 28, 28))
    return image

#Given logits and labels and var_list to optimize, returns optimizer and some statistics
def optimize(logits, labels, var_list, learning_rate = 0.01, additional_var = None):
    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels,1)
    accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    if additional_var is None: #Simultaneously minimizes additional_var if provided
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy, var_list=var_list)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy + additional_var, var_list=var_list)
    return (optimizer, accuracy, cross_entropy)




