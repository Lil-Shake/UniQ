import numpy as np

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def intersect_2d_topk(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    x2_0 = np.column_stack((x2[:,0], x2[:,1], x2[:,4]))
    x2_1 = np.column_stack((x2[:,0], x2[:,2], x2[:,4]))
    x2_2 = np.column_stack((x2[:,0], x2[:,3], x2[:,4]))
    if x1.shape[1] != x2_0.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    result_1 = np.expand_dims((x1[..., None] == x2_0.T[None, ...]).all(1), axis=-1) # [m1, m2, 3] -> [m1, m2]
    result_2 = np.expand_dims((x1[..., None] == x2_1.T[None, ...]).all(1), axis=-1)
    result_3 = np.expand_dims((x1[..., None] == x2_2.T[None, ...]).all(1), axis=-1)
    result = np.concatenate((result_1, result_2, result_3), axis=-1)
    # print(f'result_1:{result_1.shape}')
    # print(f'result shape:{result.shape}')
    # print(f'result:{result}')
    tmp = result.any(-1)
    # print(f'tmp:{tmp}')
    # print(f'tmp:{tmp.shape}')
    
    res = result.any(-1)
    return res

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
