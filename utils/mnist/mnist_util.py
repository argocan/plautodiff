import _pickle as cPickle, gzip, numpy as np, matplotlib.pyplot as plt
import os


def get_out_array_from_label(num):
    return [0.0 if x!=num else 1.0 for x in range(0,10)]


def get_label_from_out_array(arr):
    acc = 0
    for i in range(0, 10):
        acc = acc + i * arr[i]
    return acc


def get_out_array(out):
    return np.array([get_out_array_from_label(x) for x in out])


def get_label_from_result(result):
    return np.argmax(result)


def load_data_set():
    print("Loading data set...")
    print(os.path.dirname(__file__))
    f = gzip.open(os.path.dirname(__file__) + '/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()
    minp = train_set[0]
    mout = get_out_array(train_set[1])
    mtest = test_set[0]
    mtestout = get_out_array(test_set[1])
    print(minp.shape)
    print(mout.shape)
    print(mtest.shape)
    print(mtestout.shape)
    print("Data set loaded...")
    return minp, mout, mtest, mtestout

def score_result(actual, expected):

    def error_ratio(yout):
        errors = [0 if x == 0 else 1 for x in yout]
        num_errors = np.sum(errors)
        return np.true_divide(num_errors, expected.shape[0]) * 100

    result = np.argmax(actual, axis=1)
    score = result - np.argmax(expected, axis=1)
    return error_ratio(score)


def draw_image(a, title="Untitled"):
    i = np.squeeze(a[0], axis=2) if len(a.shape) == 4 else a
    fig = plt.figure()
    fig.suptitle(title)
    plt.imshow(i, cmap='gray')

def draws_show():
    plt.show()

