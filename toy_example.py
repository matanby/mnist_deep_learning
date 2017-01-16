import numpy as np
import numpy.matlib as matlib
from scipy import signal
import matplotlib.pyplot as plt

# Hyper-parameters
X_d = 4
NOISE = .01
LAMB = 0.0
BETA1 = .8
BETA2 = .9
EPS = 1e-4
BATCH_SIZE = 50
TRAINING_EPOCHS = 1000
LEARNING_RATE = lambda e: .001 if e < 300 else 0.0001
TRAIN_SET_SIZE = 2000
TEST_SET_SIZE = 200

# Labeling functions
_W = np.round(5 * (np.random.random((X_d, X_d)) - 0.5))
_W[abs(_W) > 3] = 0
f = [
    lambda x: np.sum(np.dot(x, _W), axis=1),
    lambda x: (x[:, 0] * _W[0, 0] + x[:, 1] * _W[1, 1]) * (x[:, 2] * _W[2, 2] + x[:, 1] * _W[3, 3]),
    lambda x: np.log(np.sum(np.exp(np.dot(x, _W)), axis=1))
]

# Train/Test data generation
x_train = 5 * (np.random.random((TRAIN_SET_SIZE, X_d)) - .5)
x_test = 5 * (np.random.random((TEST_SET_SIZE, X_d)) - .5)
y_train = [f[i](x_train) * (1 + np.random.randn(x_train.shape[0]) * NOISE) for i in range(len(f))]
y_test = [f[i](x_test) * (1 + np.random.randn(x_test.shape[0]) * NOISE) for i in range(len(f))]

# Allocate a structure for learning accuracy and models
models = {
    'lin': {
        i: {
            'w': None,
            'predict': lambda x: np.dot(models['lin'][i]['w'], x.T),
            'train_loss': [],
            'test_loss': [],
        }
        for i in range(len(f))
    },
    'cnn': {
        i: {
            'predict': lambda x: forward(models['cnn'][i], x)['p'],
            'train_loss': [],
            'test_loss': [],
        }
        for i in range(len(f))
    }
}


# plots charts
def plot_charts(model):
    for fi in model:
        plt.subplot(3, 2, fi * 2 + 1)
        l = len(model[fi]['train_loss'])
        plt.plot(np.arange(l), model[fi]['train_loss'], np.arange(l), model[fi]['test_loss'], lw=2)
        plt.ylim([0, 200])
        plt.subplot(3, 2, fi * 2 + 2)
        # plt.scatter(np.dot(model[fi]['w'], x_test.T), y_test[fi])
        plt.scatter(forward(model[fi], x_test)['p'], y_test[fi])
        # plt.scatter(model[fi]['predict'](x_test), y_test[fi])
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        # plt.axis('equal')
    plt.show()


# learn a toy CNN
def forward(model, x):
    """Fill a dict with forward pass variables"""
    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(0, signal.convolve2d(x, [model['w1']], mode='same'))
    fwd['o2'] = np.maximum(0, signal.convolve2d(x, [model['w2']], mode='same'))
    fwd['m'] = np.array([
        np.maximum(fwd['o1'].T[0], fwd['o1'].T[1]),
        np.maximum(fwd['o1'].T[2], fwd['o1'].T[3]),
        np.maximum(fwd['o2'].T[0], fwd['o2'].T[1]),
        np.maximum(fwd['o2'].T[2], fwd['o2'].T[3])
    ])
    fwd['p'] = model['u'].dot(fwd['m'])
    return fwd


def backprop(model, y, fwd):
    """Return the derivative of the loss w.r.t. model"""

    u = model['u']
    w1 = model['w1']
    w2 = model['w2']

    # x = fwd['x']
    # o1 = fwd['o1']
    # o2 = fwd['o2']
    # m = fwd['m']
    # p = fwd['p']
    # y = y

    x = fwd['x'].mean(axis=0)
    o1 = fwd['o1'].mean(axis=0).T
    o2 = fwd['o2'].mean(axis=0).T
    m = fwd['m'].mean(axis=1)
    p = fwd['p'].mean(axis=0)
    y = y.mean(axis=0)

    dl_du = 2 * (p - y) * m

    dl_dp = 2 * (p - y)

    dp_dm1 = u.T[:2]
    dp_dm2 = u.T[2:]

    dm1_do1 = np.zeros([2, 4])
    dm1_do1[0][0] = 1 if o1[0] >= o1[1] else 0
    dm1_do1[0][1] = 1 if o1[1] > o1[0] else 0
    dm1_do1[1][2] = 1 if o1[2] >= o1[3] else 0
    dm1_do1[1][3] = 1 if o1[3] > o1[2] else 0

    dm2_do2 = np.zeros([2, 4])
    dm2_do2[0][0] = 1 if o2[0] >= o2[1] else 0
    dm2_do2[0][1] = 1 if o2[1] > o2[0] else 0
    dm2_do2[1][2] = 1 if o2[2] >= o2[3] else 0
    dm2_do2[1][3] = 1 if o2[3] > o2[2] else 0

    do1_dw1 = np.zeros([4, 3])
    do1_dw1[0][0] = 0
    do1_dw1[0][1] = x[0]
    do1_dw1[0][2] = x[1]

    do1_dw1[1][0] = x[0]
    do1_dw1[1][1] = x[1]
    do1_dw1[1][2] = x[2]

    do1_dw1[2][0] = x[1]
    do1_dw1[2][1] = x[2]
    do1_dw1[2][2] = x[3]

    do1_dw1[3][0] = x[2]
    do1_dw1[3][1] = x[3]
    do1_dw1[3][2] = 0

    do2_dw2 = np.zeros([4, 3])
    do2_dw2[0][0] = 0
    do2_dw2[0][1] = x[0]
    do2_dw2[0][2] = x[1]

    do2_dw2[1][0] = x[0]
    do2_dw2[1][1] = x[1]
    do2_dw2[1][2] = x[2]

    do2_dw2[2][0] = x[1]
    do2_dw2[2][1] = x[2]
    do2_dw2[2][2] = x[3]

    do2_dw2[3][0] = x[2]
    do2_dw2[3][1] = x[3]
    do2_dw2[3][2] = 0

    dl_dw1 = dl_dp * dp_dm1.dot(dm1_do1).dot(do1_dw1)[::-1]
    dl_dw2 = dl_dp * dp_dm2.dot(dm2_do2).dot(do2_dw2)[::-1]

    dl_dw1 = dl_dw1.reshape((1, 3))
    dl_dw2 = dl_dw2.reshape((1, 3))
    dl_du = dl_du.reshape((1, 4))

    dl_dtheta = np.hstack((dl_dw1, dl_dw2, dl_du))
    return dl_dtheta


# learn a linear model
def train_linear():
    for fi in models['lin']:
        m_t, v_t = 0, 0
        model = models['lin'][fi]
        model['w'] = np.zeros(X_d)
        for ei in range(TRAINING_EPOCHS):
            for bi in range(0, len(y_train[fi]), BATCH_SIZE):
                idx = np.random.randint(0, len(y_train[fi]), BATCH_SIZE)
                xx, yy = x_train[idx, :], y_train[fi][idx]
                p = model['w'].dot(xx.T)
                l = np.sum((p - yy)**2 + LAMB * np.linalg.norm(model['w'], ord=2)**2) / BATCH_SIZE
                # dl_dp = ?
                a = matlib.repmat(2 * (p - yy), 4, 1).T
                b = np.multiply(a, xx) + 2 * LAMB * np.linalg.norm(model['w'], ord=2)
                dl_dw = np.mean(b, axis=0)
                m_t = BETA1 * m_t + (1 - BETA1) * dl_dw
                v_t = BETA2 * v_t + (1 - BETA2) * dl_dw ** 2
                model['w'] -= LEARNING_RATE(ei) * (m_t / (1 - BETA1)) / (np.sqrt(v_t / (1 - BETA2)) + EPS)
                model['train_loss'].append(l)
                model['test_loss'].append(np.mean((y_test[fi] - np.dot(x_test, model['w'])) ** 2))


def train_cnn():
    for fi in models['cnn']:
        m_t, v_t = 0, 0
        model = models['cnn'][fi]
        theta = .1 * (np.random.randn(10) - .5)
        model['w1'] = theta[:3]
        model['w2'] = theta[3:6]
        model['u'] = theta[6:]
        for ei in range(TRAINING_EPOCHS):
            for bi in range(0, len(y_train[fi]), BATCH_SIZE):
                idx = np.random.randint(0, len(y_train[fi]), BATCH_SIZE)
                xx, yy = x_train[idx, :], y_train[fi][idx]
                fwd = forward(model, xx)
                l = np.sum((fwd['p'] - yy) ** 2) / BATCH_SIZE
                dl_dtheta = backprop(model, yy, fwd).reshape(10, )
                dl_dtheta += LAMB * theta
                m_t = BETA1 * m_t + (1 - BETA1) * dl_dtheta
                v_t = BETA2 * v_t + (1 - BETA2) * dl_dtheta ** 2
                theta -= LEARNING_RATE(ei) * (m_t / (1 - BETA1)) / (np.sqrt(v_t / (1 - BETA2)) + EPS)
                model['w1'] = theta[:3]
                model['w2'] = theta[3:6]
                model['u'] = theta[6:]
                model['train_loss'].append(l)
                model['test_loss'].append(np.mean((y_test[fi] - forward(model, x_test)['p']) ** 2))
            print('fi:', fi, 'epoch:', ei, 'loss:', model['test_loss'][-1])


if __name__ == '__main__':
    # train_linear()
    # plot_charts(models['lin'])
    train_cnn()
    plot_charts(models['cnn'])

    # model = {
    #     'u': np.array([-1, 1, -1, 1]),
    #     'w1': np.array([1, -1, 2]),
    #     'w2': np.array([0, 2, 1]),
    # }
    #hpv
    # fwd = {
    #     'm': np.array([5, 6, 2, 3]).T,
    #     'p': np.array([2]).T,
    #     'x': np.array([1, -1, 2, -2]).T,
    #     'o1': np.array([0, 5, 0, 6]).T,
    #     'o2': np.array([2, 0, 3, 0]).T,
    # }
    #
    # y = np.array([3])
    # backprop(model, y, fwd)
