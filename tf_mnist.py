from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from scipy import ndimage


def train_linear():
    # Hyper-parameters
    BATCH_SIZE = 100
    TRAINING_EPOCHS = 10  # 50
    LEARNING_RATE = 0.01
    DISPLAY_STEP = 1

    # Network Parameters
    N_INPUT = mnist.train.images[0].shape[0]  # MNIST data input (img shape: 28*28)
    N_CLASSES = 10  # MNIST total classes (0-9 digits)

    # tf Graph Input
    x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    # Store layers weight & bias
    W = tf.Variable(tf.zeros([N_INPUT, N_CLASSES]), name='weights')
    b = tf.Variable(tf.zeros([N_CLASSES]), name='bias')

    # Define the prediction
    pred = tf.matmul(x, W) + b

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Define the accuracy metric
    true_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': tf.placeholder(tf.float32),
        'accuracy': accuracy,
        'cost': cost,
        'optimizer': optimizer
    }

    return optimize_model(model, BATCH_SIZE, DISPLAY_STEP, TRAINING_EPOCHS)


def train_mlp():
    # Hyper-parameters
    BATCH_SIZE = 100
    TRAINING_EPOCHS = 150
    LEARNING_RATE = 0.001
    DISPLAY_STEP = 1

    # Network Parameters
    N_HIDDEN_1 = 256  # 1st layer number of features
    N_HIDDEN_2 = 256  # 2nd layer number of features
    N_INPUT = mnist.train.images[0].shape[0]  # MNIST data input (img shape: 28*28)
    N_CLASSES = 10  # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1])),
        'h2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2])),
        'out': tf.Variable(tf.random_normal([N_HIDDEN_2, N_CLASSES]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([N_HIDDEN_1])),
        'b2': tf.Variable(tf.random_normal([N_HIDDEN_2])),
        'out': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    # Create model:
    # Hidden layer #1 with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer #2 with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    pred = tf.matmul(layer_2, weights['out']) + biases['out']

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Define the accuracy metric
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': tf.placeholder(tf.float32),
        'accuracy': accuracy,
        'cost': cost,
        'optimizer': optimizer
    }

    return optimize_model(model, BATCH_SIZE, DISPLAY_STEP, TRAINING_EPOCHS)


def train_cnn():
    # Hyper-parameters
    BATCH_SIZE = 100
    TRAINING_EPOCHS = 10  # 2000
    LEARNING_RATE = 1e-4
    DISPLAY_STEP = 1

    # Network Parameters
    N_INPUT = mnist.train.images[0].shape[0]  # MNIST data input (img shape: 28*28)
    IMG_DIM = int(np.sqrt(N_INPUT))  # MNIST data input (img shape: 28*28)
    N_CLASSES = 10  # MNIST total classes (0-9 digits)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # tf Graph Input
    x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    x_image = tf.reshape(x, [-1, IMG_DIM, IMG_DIM, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, N_CLASSES])
    b_fc2 = bias_variable([N_CLASSES])

    pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': keep_prob,
        'accuracy': accuracy,
        'cost': cost,
        'optimizer': optimizer
    }

    return optimize_model(model, BATCH_SIZE, DISPLAY_STEP, TRAINING_EPOCHS)


def plot_charts(*models):
    n_models = len(models)
    for mi, (name, train_loss_seq, test_loss_seq, test_accuracy_seq) in enumerate(models):
        ax = plt.subplot(2, n_models, mi + 1)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        l1, = plt.plot(train_loss_seq, lw=2, label='Train Loss')
        l2, = plt.plot(test_loss_seq, lw=2, label='Test Loss')
        plt.legend(handles=[l1, l2])
        ax = plt.subplot(2, n_models, n_models + mi + 1)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        plt.plot(test_accuracy_seq, lw=2)
    plt.show()


def optimize_model(model, batch_size, display_step, training_epochs):
    x = model['x']
    y = model['y']
    keep_prob = model['keep_prob']
    accuracy = model['accuracy']
    cost = model['cost']
    optimizer = model['optimizer']

    train_loss_seq = []
    test_loss_seq = []
    test_accuracy_seq = []

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_loss = 0.
            n_batches = int(mnist.train.num_examples / batch_size)

            # Loop over all batches
            for i in range(n_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, batch_loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                # Compute average loss
                avg_loss += batch_loss / n_batches

            test_loss = cost.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_loss_seq.append(avg_loss)
            test_loss_seq.append(test_loss)
            test_accuracy_seq.append(test_accuracy)

            # Display logs per epoch step
            if epoch % display_step == 0:
                print('Epoch: %04d cost=%.9f' % ((epoch + 1), avg_loss))

        print("Optimization Finished!")

        # Test model - calculate accuracy
        print("Accuracy:", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
    return train_loss_seq, test_loss_seq, test_accuracy_seq


# noinspection PyProtectedMember
def resize(mnist, factor):
    imsize = 28

    train = mnist.train._images
    train = train.reshape([train.shape[0], imsize, imsize])
    train = ndimage.zoom(train, (1, factor, factor))
    imsize_new = train.shape[1]
    train = train.reshape([train.shape[0], imsize_new**2])
    mnist.train._images = train

    test = mnist.test._images
    test = test.reshape([test.shape[0], imsize, imsize])
    test = ndimage.zoom(test, (1, factor, factor))
    test = test.reshape([test.shape[0], imsize_new**2])
    mnist.test._images = test

    return mnist


# noinspection PyProtectedMember
def add_noise(mnist, std):
    mnist.train._images += std * np.random.randn(*mnist.train._images.shape)
    mnist.test._images += std * np.random.randn(*mnist.test._images.shape)
    return mnist


if __name__ == '__main__':
    # Download and read the MNIST data-set
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # mnist = resize(mnist, 0.5)
    mnist = add_noise(mnist, 0.1)

    plot_charts(
        ('Linear', *train_linear()),
        ('MLP', *train_mlp()),
        ('CNN', *train_cnn()),
    )
