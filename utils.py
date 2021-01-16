import numpy as np
import scipy as sp

def _correlation2d(input:np.ndarray, filter:np.ndarray, stride:int = 1, dil:int = 0, pad:int = 0) -> np.ndarray:
    '''
    basic 2d correlation operator in1*in2. simply move throughout in1 and evaluate the local dot product with in2.


    :param in1: 3D ndarray, second input for the correlation operator. shape: CxHxW.
    :param filter: 3D ndarray, second input for the correlation operator. shape: CxHxW.
    :param stride: int, how many entries to skip (both horizontally and vertically) when moving in2 throughout in1.
                        (default: 1)
    :param dil: int, spacing between elements of in2.
                        (default: 0)
    :param pad: int, amount of zero pad for in1.
                        (default: 0)
    :return:
    '''

    c_in, h_in, w_in = input.shape
    cf, hf, wf = filter.shape

    assert c_in == cf, "ERROR: input and filters must have the same number of channels!\n" \
                     "input shape: {} filter shape: {}".format(input.shape, filter.shape)

    if not (hf<=h_in or wf<=w_in):
        Warning('filter dimesions are larger than input! may lead to unexpected behaviour.\n'
                'input shape: {} filter shape: {}'.format(input.shape, filter.shape))

    # out_w = int((w_in + 2 * pad - dil * (wf - 1) - 1)/stride) + 1
    # out_h = int((h_in + 2 * pad - dil * (hf - 1) - 1)/stride) + 1

    # np.pad defaults to zero padding, do not pad in the channnel dimension
    if pad>0:
        input = np.pad(input, ((0,0), (pad,pad), (pad, pad)))


def correlation2d(inputs:np.ndarray, kernel:np.ndarray, kwargs) -> np.ndarray:
    '''
    handle for _correlation2d() which computes the correlation between two numpy arrays.


    :param inputs: ndarray 3-4D, input(s) for the correlation operator.
    :param kernel: ndarray 4D, kernel weights of the conv layer. 4D because many filters are used in each layer.
    :param kwargs: keyworded additional parameters to pass to _correlation2d.


    :return: ndarray
    '''

    if len(inputs.shape)>3:
        output = []
        for _input in inputs:
            f_maps = []
            for filter in kernel:
                f_maps.append(_correlation2d(_input, filter, **kwargs))
            output.append(np.stack(f_maps))
    else:
        output = []
        for filter in kernel:
            output.append(_correlation2d(inputs, filter, **kwargs))

    return np.stack(output)


def _relu(input):
    mask = input <= 0
    input[mask] = 0
    # grad = np.ones_like(input)
    # grad[mask] = 0
    return input