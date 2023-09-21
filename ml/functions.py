def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    result = 1
    for i in range(len(x)):
        if len(x[i]) > i and x[i][i] != 0:
            result *= x[i][i]
    return result


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    if len(x) != len(y):
        return False

    x.sort()
    y.sort()
    for i in range(len(x)):
        if x[i] != y[i]:
            return False

    return True


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    y = []
    for i in range(1, len(x)):
        if x[i - 1] == 0:
            y.append(x[i])
    return max(y)


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x num_channels)
    coefs -- 1-d numpy array (length num_channels)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    height = len(img)
    width = len(img[0])
    numChannels = len(img[0][0])
    result = []

    for i in range(height):
        result.append([])
        for j in range(width):
            tmp = 0
            for k in range(numChannels):
                tmp += img[i][j][k] * coefs[k]
            result[i].append(tmp)

    return result


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    numbers = []
    times = []
    for i in range(len(x)):
        if len(numbers) == 0 or numbers[-1] != x[i]:
            numbers.append(x[i])
            times.append(1)
        else:
            times[-1] += 1
    return numbers, times


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    result = []
    mx = len(x)
    my = len(y)
    n = len(x[0])
    for i in range(mx):
        result.append([])
        for j in range(my):
            tmp = 0
            for k in range(n):
                tmp += (x[i][k] - y[j][k]) ** 2
            result[i].append(tmp ** 0.5)

    return result