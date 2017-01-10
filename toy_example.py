from scipy import signal
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# functions
X_d = 4
W = np.round(5 * (np.random.random((X_d, X_d)) - 0.5))
W[abs(W) > 3] = 0

y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: (x[:, 0] * W[0, 0] + x[:, 1] * W[1, 1]) * (x[:, 2] * W[2, 2] + x[:, 1] * W[3, 3]),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}

# data
X = {
    'train': 5 * (np.random.random((1000, X_d)) - .5),
    'test': 5 * (np.random.random((200, X_d)) - .5)
}

noise = .01
Y = {
    i: {
        'train': y[i](X['train']) * (1 + np.random.randn(X['train'].shape[0]) * noise),
        'test': y[i](X['test']) * (1 + np.random.randn(X['test'].shape[0]) * noise)
    }
    for i in range(len(y))
}

# set hyper-parameters, and allocate a structure for learning accuracy and models
batch = 100
lamb = 0
beta1 = .8
beta2 = .9
eps = 1e-4
epochs = 10
rate = lambda e: .001
models = {
    'lin': {i: dict(loss=dict(train=[], test=[])) for i in range(len(y))},
    'cnn': {i: dict(loss=dict(train=[], test=[])) for i in range(len(y))}
}

# learn a linear model
for fi in range(len(y)):
    m_t, v_t = 0, 0
    model = models['lin'][fi]
    model['w'] = np.zeros(X_d)
    for ei in range(epochs):
        for bi in range(0, len(Y[fi]['train']), batch):
            idx = np.random.randint(0, len(Y[fi]['train']), batch)
            xx, yy = X['train'][idx, :], Y[fi]['train'][idx]
            p = model['w'].dot(xx)
            l = (p - yy)**2 + lamb * np.linalg.norm(model['w'], ord=2)
            dl_dp = ?
            dl_dw = ?
            m_t = beta1 * m_t + (1 - beta1) * dl_dw
            v_t = beta2 * v_t + (1 - beta2) * dl_dw ** 2
            model['w'] -= rate(ei) * (m_t / (1 - beta1)) / (np.sqrt(v_t / (1 - beta2)) + eps)
            model['loss']['train'].append(l / batch)
            model['loss']['test'].append(np.mean((Y[fi]['test'] - np.dot(X['test'], model['w'])) ** 2))


# learn a toy CNN
def forward(model, x):
    """Fill a dict with forward pass variables"""
    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(0, signal.convolve2d(x, [model['w1']], mode='same'))
    fwd['o2'] = ?
    fwd['m'] = ?
    fwd['p'] = ?
    return fwd


def backprop(model, y, fwd):
    """Return the derivative of the loss w.r.t. model"""
    ...
    dl_dtheta = np.hstack((dl_dw1, dl_dw2, dl_du))
    return dl_dtheta


for fi in range(len(y)):
    m_t, v_t = 0, 0
    model = models['cnn'][fi]
    theta = .1 * (np.random.randn(10) - .5)
    model['w1'] = model['theta'][:3]
    model['w2'] = model['theta'][3:6]
    model['u'] = model['theta'][6:]
    for ei in range(epochs):
        for bi in range(0, len(Y[fi]['train']), batch):
            idx = np.random.randint(0, len(Y[fi]['train']), batch)
            xx, yy = X['train'][idx, :], Y[fi]['train'][idx]
            fwd = forward(model, xx)
            l = np.sum((fwd['p'] - yy) ** 2)
            dl_dtheta = backprop(model, yy, fwd) + lamb * theta
            m_t = beta1 * m_t + (1 - beta1) * dl_dtheta
            v_t = beta2 * v_t + (1 - beta2) * (dl_dtheta) ** 2
            theta -= rate(ei) * (m_t / (1 - beta1)) / (np.sqrt(v_t / (1 - beta2)) + eps)
            model['loss']['train'].append(l / batch)
            model['loss']['test'].append(np.mean((Y[fi]['test'] - forward(model, X['test'])['p']) ** 2))

# some plots
for i in range(len(y)):
    plt.subplot(3, 2, i * 2 + 1)
    l = len(models['cnn'][i]['loss']['train'])
    plt.plot(np.arange(l), models['cnn'][i]['loss']['train'],
             np.arange(l), models['cnn'][i]['loss']['test'], lw=2)
    plt.ylim([0, 20])
    plt.subplot(3, 2, i * 2 + 2)
    plt.scatter(forward(models['cnn'][i], X['test'])['p'], Y[i]['test'])
    plt.axis('equal')


if __name__ == '__main__':
    pass