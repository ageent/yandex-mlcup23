import numpy as np
from scipy.special import boxcox, inv_boxcox


def boxcox_func(data, inverse=False):
    data = np.array(data)

    nonzero_mask = data > 0
    transformed_data = data[nonzero_mask]

    lambda_const = -1.0
    if transformed_data.size:
        if inverse:
            transformed_data = inv_boxcox(transformed_data, lambda_const) - 1
        else:
            transformed_data = boxcox(transformed_data + 1, lambda_const)
        data[nonzero_mask] = transformed_data

    return data
