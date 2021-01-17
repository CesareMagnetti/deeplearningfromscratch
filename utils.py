import numpy as np
import scipy as sp

def correlation2d(input:np.ndarray, kernel:np.ndarray, bias:np.ndarray, stride:int = 1, padding:int = 0) -> np.ndarray:
    '''
    basic 2d correlation operator in1*in2. simply move throughout in1 and evaluate the local dot product with in2.


    :param in1: 4D ndarray, second input for the correlation operator. shape: BxCxHxW.
    :param kernel: 4D ndarray, N kernels of the conv layer. shape: NxCxHxW.
    :param bias: 1D ndarray, each kernel has its own bias term added element wise. shape: N.
    :param stride: int, how many entries to skip (both horizontally and vertically) when moving filter throughout input.
                        (default: 1)
    :param padding: int, amount of zero pad for input.
                        (default: 0)
    :return:
    '''

    B,C,H,W = input.shape # B is the batch size (number of inputs)
    Nk,Ck,Hk,Wk = kernel.shape # Nk is the number of kernels used. EACH OF THE B INPUTs GOES THROUGH ALL Nk KERNELS.

    # 1 kernel and input must have the same number of channels
    assert C == Ck, "ERROR: input and filters must have the same number of channels!"

    if not (Hk<=H or Wk<=W):
        Warning('kernel size is larger than the input! may lead to unexpected behaviour.\n'
                'input shape: {} filter shape: {}'.format(input.shape[-2:], kernel.shape[-2:]))

    # 2 get output dimensions
    Hout = int((H - Hk + 2 * padding)/stride) + 1
    Wout = int((W - Wk + 2 * padding)/stride) + 1


    # 3 pad image if necessary
    # np.pad defaults to zero padding, do not pad in the channnel dimension
    if padding>0:
        input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding, padding)), mode="constant")
        #B, C, H, W = input.shape  # B is the batch size (number of inputs)

    # 4 reshape kernels from Nk x Ck x Hk x Wk to Nk x Ck*Hk*Wk
    kernel = kernel.reshape(Nk, -1)

    # 5 transform input to a matrix whose columns are the neighborhood of the kernel at each position. Since inputs are
    # processed in batches we will stack B of such matrices together to achieve a shape of B x (Ck*Hk*Wk) x (H*W).
    # Note that this is not really a reshape because elements are repeated.
    # for a good overview of this concept see publication:
    # https://www.researchgate.net/publication/330315719_A_Uniform_Architecture_Design_for_Accelerating_2D_and_3D_CNNs_on_FPGAs

    # get relative indices of the kernel's neighbourhood (row major order)
    i = np.tile(np.repeat(np.arange(Hk), Wk), Ck)  # row index
    j = np.tile(np.arange(Wk), Ck*Hk) # column index
    k = np.repeat(np.arange(Ck), Wk*Hk) # depth index (channels)
    # neigh_index --> [k, i, j] # indexing as BxCxHxW

    # using the top left corner of the mask as the origin of the kernel, get the locations where the mask will be placed
    I = stride*np.repeat(np.arange(Hout), Wout)
    J = stride*np.tile(np.arange(Wout), Hout)

    # get the final indeces at each position I,J,K

    ii = i.reshape(-1, 1) + I.reshape(1, -1)
    jj = j.reshape(-1, 1) + J.reshape(1, -1)
    kk = k.reshape(-1, 1)

    # get the entries of the input array at the selected indeces
    transformed_input = input[:, kk, ii, jj]

    # 6 perform convolution as matrix multpilication, for each input in the batch and for each kernel used.
    # this means multiply a Nk x (Ck*Hk*Wk) matrix to a (Ck*Hk*Wk) x (Hout*Wout) matrix B times. numpy built-in
    # matrix multiplication operator @ will take care of broadcasting the matrix multiplication along the B dimension,
    # however we need to take care of broadcasting the bias in a Nkx(Hout*Wout) matrix, where each row has the bias of
    # the corresponding filter

    b = np.ones((Nk, Hout*Wout))*np.expand_dims(bias, axis=1)
    out = kernel @ transformed_input + b

    # reshape to proper form
    out = out.reshape(B, Nk, Hout, Wout)

    return out

def _relu(input):
    mask = input <= 0
    input[mask] = 0
    # grad = np.ones_like(input)
    # grad[mask] = 0
    return input