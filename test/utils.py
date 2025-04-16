import numpy as np


def random_vector(seed: int, dimension: int):
    np.random.seed(seed)
    return list(np.random.uniform(low=-1., high=1., size=(dimension,)))
