import numpy as np
import scipy.io

"""
Download 'mnist_all.mat' from
http://www.cs.nyu.edu/~roweis/data.html
"""


def parse_mnist(rank=2, mnist_path='/home/khshim/data/mnist/', num_valid=10000, seed=None):
    mnist_file = mnist_path + 'mnist_all.mat'
    mnist = scipy.io.loadmat(mnist_file)

    train = (mnist['train0'], mnist['train1'], mnist['train2'], mnist['train3'], mnist['train4'],
             mnist['train5'], mnist['train6'], mnist['train7'], mnist['train8'], mnist['train9'])
    train_data = np.vstack(train)
    train_label = np.zeros((len(train_data),), dtype='int64')
    start = 0
    for i, p in enumerate(train):
        length = len(p)
        train_label[start:start + length] = i
        start = start + length
    assert start == len(train_data)

    test = (mnist['test0'], mnist['test1'], mnist['test2'], mnist['test3'], mnist['test4'],
            mnist['test5'], mnist['test6'], mnist['test7'], mnist['test8'], mnist['test9'])
    test_data = np.vstack(test)
    test_label = np.zeros((len(test_data),), dtype='int64')
    start = 0
    for i, p in enumerate(test):
        length = len(p)
        test_label[start:start + length] = i
        start = start + length
    assert start == len(test_data)

    if seed is not None:
        np_rng = np.random.RandomState(seed)
        order = np_rng.permutation(train_data.shape[0])
    else:
        order = np.random.permutation(train_data.shape[0])

    if rank == 2:
        pass
    elif rank == 4:
        train_data = np.reshape(train_data, [-1, 1, 28, 28])
        test_data = np.reshape(test_data, [-1, 1, 28, 28])
    else:
        raise ValueError('Invalid mode')

    train_data = train_data[order]
    train_label = train_label[order]
    valid_data = train_data[:num_valid]
    valid_label = train_label[:num_valid]
    train_data = train_data[num_valid:]
    train_label = train_label[num_valid:]

    return train_data, train_label, valid_data, valid_label, test_data, test_label
