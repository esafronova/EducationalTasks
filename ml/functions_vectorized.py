import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    d = np.diag(x)
    result = np.prod(d[d != 0])
    return result


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    return np.all(np.sort(x) == np.sort(y))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    indexes = x[:-1] == 0
    after_zero = (x[1:])[indexes]
    return after_zero.max()


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x num_channels)
    coefs -- 1-d numpy array (length num_channels)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    result = coefs * img
    return result.sum(axis=2)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    bounded = np.hstack(([0], x, [0]))
    difs = np.diff(bounded)
    runs = np.arange(difs.size)[difs != 0]
    numbers = x[runs[:-1]]
    times = runs[1:] - runs[: -1]
    return numbers, times


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    mx = x.shape[0]
    my = y.shape[0]
    a = np.tile(np.expand_dims(x, axis=1), (1, my, 1))
    b = np.tile(np.expand_dims(y, axis=0), (mx, 1, 1))

    result = (a - b) ** 2
    result = result.sum(axis=2) ** 0.5
    return result