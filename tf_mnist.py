from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from scipy import ndimage


mnist = None
NOISE_STD = 0.4
REDUCE_FACTOR = 0.5


def static_learning_rate(learning_rate):
    return lambda loss_diff, prev_lr, epoch: learning_rate


def dynamic_learning_rate(loss_diff, prev_lr, epoch):
    return prev_lr / 5 if loss_diff / prev_lr < 0.1 else prev_lr


def dynamic_learning_rate2(loss_diff, prev_lr, epoch):
    return 1/((epoch+1) * 100)


def train_linear(batch_size=100, training_epochs=10, learning_rate=5e-4, display_step=1):
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
    learning_rate_ph = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate_ph).minimize(cost)

    # Define the accuracy metric
    true_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': tf.placeholder(tf.float32),
        'accuracy': accuracy,
        'learning_rate_ph': learning_rate_ph,
        'cost': cost,
        'optimizer': optimizer
    }

    return optimize_model(model, batch_size, display_step, training_epochs, static_learning_rate(learning_rate))


def train_mlp(batch_size=100, training_epochs=10, learning_rate=5e-3, display_step=1):
    # Network Parameters
    N_HIDDEN_1 = 256  # 1st layer number of features
    N_HIDDEN_2 = 256  # 2nd layer number of features
    N_INPUT = mnist.train.images[0].shape[0]  # MNIST data input (img shape: 28*28)
    N_CLASSES = 10  # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    # Define layers weight & bias
    w1 = tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_1]))
    w2 = tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2]))
    w_out = tf.Variable(tf.random_normal([N_HIDDEN_2, N_CLASSES]))

    b1 = tf.Variable(tf.random_normal([N_HIDDEN_1]))
    b2 = tf.Variable(tf.random_normal([N_HIDDEN_2]))
    b_out = tf.Variable(tf.random_normal([N_CLASSES]))

    # Hidden layer #1 with RELU activation
    layer_1 = tf.add(tf.matmul(x, w1), b1)
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer #2 with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    pred = tf.add(tf.matmul(layer_2, w_out), b_out)

    # Define loss and optimizer
    learning_rate_ph = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate_ph).minimize(cost)

    # Define the accuracy metric
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': tf.placeholder(tf.float32),
        'accuracy': accuracy,
        'learning_rate_ph': learning_rate_ph,
        'cost': cost,
        'optimizer': optimizer
    }

    return optimize_model(model, batch_size, display_step, training_epochs, static_learning_rate(learning_rate))


def train_cnn(batch_size=100, training_epochs=10, learning_rate=1e-2, display_step=1, use_xavier=False):
    # Network Parameters
    N_INPUT = mnist.train.images[0].shape[0]  # MNIST data input (img shape: 28*28)
    IMG_DIM = int(np.sqrt(N_INPUT))  # MNIST data input (img shape: 28*28)
    N_CLASSES = 10  # MNIST total classes (0-9 digits)

    def weight_variable(name, shape):
        if use_xavier:
            return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(name, shape):
        if use_xavier:
            return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # tf Graph Input
    x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    x_image = tf.reshape(x, [-1, IMG_DIM, IMG_DIM, 1])

    W_conv1 = weight_variable('W_conv1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b_conv1', [32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable('W_conv2', [5, 5, 32, 64])
    b_conv2 = bias_variable('b_conv2', [64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_shape = h_pool2.get_shape()
    new_dim = int(h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3])

    W_fc1 = weight_variable('W_fc1', [new_dim, 1024])
    b_fc1 = bias_variable('b_fc1', [1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable('W_fc2', [1024, N_CLASSES])
    b_fc2 = bias_variable('b_fc2', [N_CLASSES])

    pred = tf.matmul(h_fc1, W_fc2) + b_fc2

    learning_rate_ph = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate_ph).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': tf.placeholder(tf.float32),
        'accuracy': accuracy,
        'learning_rate_ph': learning_rate_ph,
        'cost': cost,
        'optimizer': optimizer
    }

    return optimize_model(model, batch_size, display_step, training_epochs, static_learning_rate(learning_rate))


def train_super_cnn(batch_size=100, training_epochs=10, learning_rate_func=dynamic_learning_rate, display_step=1, use_xavier=True):
    # Network Parameters
    N_INPUT = mnist.train.images[0].shape[0]  # MNIST data input (img shape: 28*28)
    IMG_DIM = int(np.sqrt(N_INPUT))  # MNIST data input (img shape: 28*28)
    N_CLASSES = 10  # MNIST total classes (0-9 digits)

    def weight_variable(name, shape):
        if use_xavier:
            return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(name, shape):
        if use_xavier:
            return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            return tf.Variable(tf.constant(0.1, shape=shape))

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # tf Graph Input
    x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    x_image = tf.reshape(x, [-1, IMG_DIM, IMG_DIM, 1])

    W_conv1 = weight_variable('W_conv1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b_conv1', [32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable('W_conv2', [5, 5, 32, 64])
    b_conv2 = bias_variable('b_conv2', [64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_shape = h_pool2.get_shape()
    new_dim = int(h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3])

    W_fc1 = weight_variable('W_fc1', [new_dim, 1024])
    b_fc1 = bias_variable('b_fc1', [1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable('W_fc2', [1024, N_CLASSES])
    b_fc2 = bias_variable('b_fc2', [N_CLASSES])

    pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    learning_rate_ph = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate_ph).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = {
        'x': x,
        'y': y,
        'keep_prob': keep_prob,
        'accuracy': accuracy,
        'learning_rate_ph': learning_rate_ph,
        'cost': cost,
        'optimizer': optimizer
    }

    res = optimize_model(model, batch_size, display_step, training_epochs, learning_rate_func)
    tf.reset_default_graph()
    return res


def optimize_model(model, batch_size, display_step, training_epochs, learning_rate_func):
    x = model['x']
    y = model['y']
    keep_prob = model['keep_prob']
    accuracy = model['accuracy']
    learning_rate_ph = model['learning_rate_ph']
    cost = model['cost']
    optimizer = model['optimizer']

    train_loss_seq = []
    test_loss_seq = []
    test_accuracy_seq = []

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        lr = 2e-3
        test_loss_diff = 100
        test_loss_prev = 100

        for epoch in range(training_epochs):
            avg_loss = 0.
            n_batches = int(mnist.train.num_examples / batch_size)

            # Get the current learning-rate
            lr = learning_rate_func(test_loss_diff, lr, epoch)

            # Loop over all batches
            for i in range(n_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, batch_loss = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, learning_rate_ph: lr, keep_prob: 0.5})

                # Compute average loss
                avg_loss += batch_loss / n_batches

            test_loss = cost.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            test_loss_diff = test_loss_prev - test_loss
            test_loss_prev = test_loss
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_loss_seq.append(avg_loss)
            test_loss_seq.append(test_loss)
            test_accuracy_seq.append(test_accuracy)

            # Display logs per epoch step
            if epoch % display_step == 0:
                print('Epoch: %04d cost=%.9f, accuracy=%.4f' % ((epoch + 1), avg_loss, test_accuracy))

        print("Optimization Finished!")

        # Test model - calculate accuracy
        print("Accuracy:", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
    return train_loss_seq, test_loss_seq, test_accuracy_seq


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
        plt.ylim(0, 1)
        plt.plot(test_accuracy_seq, lw=2)
    plt.show()


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


def compare_parameters():
    print('comparing different batch sizes')
    batch_sizes = [5, 10, 25, 50, 100, 200]
    batch_size_results = {}
    for batch_size in batch_sizes:
        _, _, test_accuracy_set = train_cnn(batch_size=batch_size)
        accuracy = test_accuracy_set[-1]
        batch_size_results[batch_size] = accuracy
        print('batch size:', batch_size, 'accuracy:', accuracy)

    print('comparing different weights initialization options')
    init_opts_results = {}
    for use_xavier in [False, True]:
        _, _, test_accuracy_set = train_cnn(use_xavier=use_xavier)
        accuracy = test_accuracy_set[-1]
        init_opts_results[use_xavier] = accuracy
        print('init option:', use_xavier, 'accuracy:', accuracy)


def single_model_all_datasets(model_name):
    global mnist

    if model_name == 'linear':
        train_model = train_linear
    elif model_name == 'mlp':
        train_model = train_mlp
    elif model_name == 'cnn':
        train_model = train_cnn
    else:
        train_model = train_super_cnn

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    m1 = train_model()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = resize(mnist, REDUCE_FACTOR)
    m2 = train_model()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = add_noise(mnist, NOISE_STD)
    m3 = train_model()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = resize(mnist, REDUCE_FACTOR)
    mnist = add_noise(mnist, NOISE_STD)
    m4 = train_model()

    plot_charts(
        ('Original MNIST', *m1),
        ('Reduced MNIST', *m2),
        ('Noisy MNIST', *m3),
        ('Reduced Noisy MNIST', *m4),
    )


def show_sample_images():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    ax = plt.subplot(1, 4, 1)
    ax.set_title('Original')
    plt.axis('off')
    dim = np.sqrt(mnist.train.images[0].shape[0])
    plt.imshow(mnist.train.images[0].reshape(dim, dim))

    ax = plt.subplot(1, 4, 2)
    ax.set_title('Reduced')
    plt.axis('off')
    mnist = resize(mnist, REDUCE_FACTOR)
    dim = np.sqrt(mnist.train.images[0].shape[0])
    plt.imshow(mnist.train.images[0].reshape(dim, dim))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    ax = plt.subplot(1, 4, 3)
    ax.set_title('Noisy')
    plt.axis('off')
    mnist = add_noise(mnist, NOISE_STD)
    dim = np.sqrt(mnist.train.images[0].shape[0])
    plt.imshow(mnist.train.images[0].reshape(dim, dim))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    ax = plt.subplot(1, 4, 4)
    ax.set_title('Reduced + Noisy')
    plt.axis('off')
    mnist = resize(mnist, REDUCE_FACTOR)
    mnist = add_noise(mnist, NOISE_STD)
    dim = np.sqrt(mnist.train.images[0].shape[0])
    plt.imshow(mnist.train.images[0].reshape(dim, dim))

    plt.show()


def compare_num_of_parameters_accuracy():
    # Download and read the MNIST data-set
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = resize(mnist, REDUCE_FACTOR)
    mnist = add_noise(mnist, NOISE_STD)

    plot_charts(
        ('MLP 16', *train_mlp(16)),
        ('MLP 32', *train_mlp(32)),
        ('MLP 64', *train_mlp(64)),
        ('MLP 128', *train_mlp(128)),
        ('MLP 256', *train_mlp(256)),
    )

if __name__ == '__main__':
    # Download and read the MNIST data-set
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = resize(mnist, REDUCE_FACTOR)
    mnist = add_noise(mnist, NOISE_STD)

    show_sample_images()
    compare_num_of_parameters_accuracy()
    single_model_all_datasets('super-cnn')
    compare_parameters()

    plot_charts(
        ('Linear', *train_linear()),
        ('MLP', *train_mlp(learning_rate=2e-3)),
        ('CNN', *train_cnn(learning_rate=2e-2)),
        ('Super-CNN', *train_super_cnn()),
    )

