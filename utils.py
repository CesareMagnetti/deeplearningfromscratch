import numpy as np
from matplotlib import pyplot as plt


def show_tensor(tensor, normalize = False, cmap = "gray"):
    if normalize:
        tensor /= tensor.max()
    plt.imshow(tensor.transpose(1, 2, 0), cmap=cmap)
    plt.title("original image")
    plt.axis("off")
    plt.show()


def show_batch_of_tensors(batch, ncol=5, normalize = False, cmap = "gray"):
    B, C, H, W = batch.shape
    extra = 0
    if not B % ncol == 0:
        Warning('inconsistent dimensions: {} samples to format using {} columns. ugly output'.format(B, ncol))
        extra+=1

    fig, axs = plt.subplots(int(B / ncol) + extra, ncol)

    for (tensor, ax) in zip(batch, axs.ravel()[:B]):
        if normalize:
            tensor /= tensor.max()
        ax.imshow(tensor.transpose(1, 2, 0), cmap=cmap)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_neighborhood_indeces(input_shape:tuple, kernel_shape:tuple, stride:int, padding:int) -> tuple:

    '''
    :param input_shape: tuple, shape of input array from which we want to extract indeces of all possible kernel neighbours
    :param kernel_shape: tuple, shape of the kernel(s) that we want to convolve with the input
    :param stride: int, spacing between possible kernel positions.
    :param padding: int, number of pixels used to pad the image

    :return: (ii, jj, kk)
             tuple of indeces corresponding to all possible neighborhoods given the above inputs
    '''

    B,C,H,W = input_shape # B is the batch size (number of inputs)
    Nk,Ck,Hk,Wk = kernel_shape # Nk is the number of kernels used. EACH OF THE B INPUTs GOES THROUGH ALL Nk KERNELS.

    # get output dimensions
    Hout = int((H - Hk + 2 * padding) / stride) + 1
    Wout = int((W - Wk + 2 * padding) / stride) + 1

    # get relative indices of the kernel's neighbourhood (row major order)
    i = np.tile(np.repeat(np.arange(Hk), Wk), Ck)  # row index
    j = np.tile(np.arange(Wk), Ck*Hk) # column index
    k = np.repeat(np.arange(Ck), Wk*Hk) # depth index (channels)
    # neigh_index --> [k, i, j] # indexing as BxCxHxW

    # using the top left corner of the mask as the origin of the kernel, get the locations where the mask will be placed
    I = stride*np.repeat(np.arange(Hout), Wout)
    J = stride*np.tile(np.arange(Wout), Hout)

    # get the final indeces of kernel's neighborhood at each position I,J,K in 2D arrays of indeces
    ii = i.reshape(-1, 1) + I.reshape(1, -1)
    jj = j.reshape(-1, 1) + J.reshape(1, -1)
    kk = k.reshape(-1, 1)

    return (ii, jj, kk)

def image2neighbourhoodMatrix(input: np.ndarray, kernel_shape:tuple, stride:int, padding:int)-> np.ndarray:

    '''
    # transform input to a matrix whose columns are the neighborhood of the kernel at each position. Since inputs are
    # processed in batches we will stack B of such matrices together to achieve a shape of B x (Ck*Hk*Wk) x (H*W).
    # Note that this is not really a reshape because elements are repeated.
    # for a good overview of this concept see publication:
    # https://www.researchgate.net/publication/330315719_A_Uniform_Architecture_Design_for_Accelerating_2D_and_3D_CNNs_on_FPGAs

    :param input: 4D np.ndarray, input (i.e. image/batch of images) to the conv layer. shape:B x C x H x W
    :param kernel: tuple (Nk,Ck,Hk,Wk), shape of kernel weights of the conv layer. note that Ck == C.
    :param stride: int, spacing between possible kernel positions.
    :param padding: int, number of pixels used to pad the image.

    :return: (transformed_input, transformed_kernel)
             tuple of the input and kernel transformed appropriately to perform convolution as matrix
             multpiplication.
    '''

    B,C,H,W = input.shape # B is the batch size (number of inputs)
    Nk,Ck,Hk,Wk = kernel_shape # Nk is the number of kernels used. EACH OF THE B INPUTs GOES THROUGH ALL Nk KERNELS.

    # pad image if necessary
    # np.pad defaults to zero padding, do not pad in the channnel dimension
    if padding>0:
        input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding, padding)))

    # get indeces of all possible neighours
    ii, jj, kk = get_neighborhood_indeces((B,C,H,W), kernel_shape, stride, padding)
    # get the entries of the input array at the selected indeces
    # neigh_index --> [kk, ii, jj] # indexing as BxCxHxW
    transformed_input = input[:, kk, ii, jj]
    # the above will be a Bx(Ck*Hk*Wk)x(H*W), reshape to (Ck*Hk*Wk)x(B*H*W)
    transformed_input = transformed_input.transpose(1, 2, 0).reshape(Hk*Wk*C, -1)

    return transformed_input

def neighborhoodMatrix2image(transformed_input:np.ndarray, input_shape:tuple, kernel_shape:tuple,
                             stride:int, padding:int) -> np.ndarray:

    '''
    basically an inverse function to image2neighbourhoodMatrix(), where we transform the input back to its
    original form. this is useful when computing the gradients of the convolutional layer given that we are transforming
    the convolution to a matrix multiplication.

    :param transformed_input: 2D np.ndarray, first output of image2neighbourhoodMatrix(). shape: (Ck*Hk*Wk) x (H*W*B)
    :param input_shape: tuple, (B,C,H,W) shape of the input to the conv layer
    :param kernel_shape: tuple, (Nk,Ck,Hk,Wk) shape of the kernel weights of the conv layer
    :param stride: int, spacing between possible kernel positions.
    :param padding: int, number of pixels used to pad the image.

    :return: input
             converts transformed_input to its original shape
    '''

    B,C,H,W = input_shape # B is the batch size (number of inputs)
    Nk,Ck,Hk,Wk = kernel_shape # Nk is the number of kernels used. EACH OF THE B INPUTs GOES THROUGH ALL Nk KERNELS.

    # input to the conv layer may have been padded, recover a zero-mask of the padded input
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    input_padded = np.zeros((B, C, H_padded, W_padded), dtype=transformed_input.dtype)

    # we then used a particular set of indices to transform 4D input into 2D matrix with each neighborhood
    # in each column, recover that
    ii, jj, kk = get_neighborhood_indeces(input_shape, kernel_shape, stride, padding)

    # reshape transformed_input so that feature maps are in a separate dimension
    input = transformed_input.reshape(C * Hk * Wk, -1, B).transpose(2, 0, 1)

    # populate the input image with the corresponding entries
    np.add.at(input_padded, (slice(B), kk, ii, jj), input)

    # we only need the gradients for the original image, hence discard padded pixels if any:
    if padding != 0:
        input = input_padded[:, :, padding:-padding, padding:-padding]
    else:
        input = input_padded

    return input

def correlation2d(input:np.ndarray, kernel:np.ndarray, bias:np.ndarray, stride:int = 1, padding:int = 0) -> tuple:

    '''
    fast 2D correlation as matrix multiplication, see paper:
    https://www.researchgate.net/publication/330315719_A_Uniform_Architecture_Design_for_Accelerating_2D_and_3D_CNNs_on_FPGAs


    :param in1: 4D np.ndarray, second input for the correlation operator. shape: BxCxHxW.
    :param kernel: 4D np.ndarray, N kernels of the conv layer. shape: NxCxHxW.
    :param bias: 1D np.ndarray, each kernel has its own bias term added element wise. shape: N.
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

    # get output dimensions
    Hout = int(np.floor((H - Hk + 2 * padding) / stride + 1))
    Wout = int(np.floor((W - Wk + 2 * padding) / stride + 1))

    # perform convolution as matrix multpilication, and add bias of each kernel
    transformed_kernel = kernel.reshape(Nk, -1) # reshape kernels from Nk x Ck x Hk x Wk to Nk x Ck*Hk*Wk
    transformed_input = image2neighbourhoodMatrix(input, kernel.shape, stride, padding) # reshape input from B x C x H x W to (Ck*Hk*Wk)x(B*H*W)
    out = transformed_kernel @ transformed_input + bias

    # reshape to proper form
    out = out.reshape(Nk, Hout, Wout, B).transpose(3,0,1,2)

    # return cache to compute derivatives in backward pass
    cache = {"input": input, "transformed_input":transformed_input}

    return out, cache