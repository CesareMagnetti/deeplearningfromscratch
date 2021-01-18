import utils
import numpy as np

class Module():
    def __init__(self):
        self.weights = None
        self.bias = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function not yet implemented!')

    def bwd(self, *args, **kwargs):
        raise NotImplementedError('bwd (backward implementation) method not yet implemented!'
                                  )
    def backward(self, dout):
        return self.bwd(dout, self.cache)

    def __call__(self, *args, **kwargs):
        out, self.cache = self.forward(*args, **kwargs)
        return out.transpose(3,0,1,2) # due to how we performed forward pass



class Linear(Module):
    '''
    implementation of a linear layer which simply implements W*input + bias and evaluates gradients once called pass
    '''
    def __init__(self, in_features: int, out_features: int, store_grad:bool = False):
        super(Linear, self).__init__()
        self.weights = np.random.rand(in_features, out_features)
        # bias is initialised to zero
        self.bias = np.zeros(out_features)

    def forward(self, inputs: np.ndarray) -> tuple:
        out = inputs @ self.weights + self.bias
        cache = inputs, self.weights
        return (out, cache)

    def bwd(self, Dout:np.ndarray, cache:tuple) -> tuple:
        '''
        function to perform backward pass on linear layer
        :param Dout: np.ndarray, differential on the output of the input layer. shape: BxN
        :param cache: tuple, containes cached variables during forward pass for easy differentiation
        :return: (dW, db, dX) tuple of partial derivatives wrt to parameters W (weights), b (bias) and input X
        '''
        input, W = cache

        # partials are trivial since linear layer simply performs XW+b, note that b is broadcasted element-wise and that
        # this can be a general inner layer, hence derivatives depend on Dout due to chain rule.
        DW = input.T @ Dout #d[XW+b}/dW = X then inner product with Dout due to chain rule.
        Db = np.sum(Dout, axis=0) #d[XW+b]/db = 1 then multiply by Dout due to chain rule.
                                  # sum over first dimension because b was broadcasted element-wise
        DX = Dout @ W.T #d[XW+b}/dX = W then inner product with Dout due to chain rule.

        return (DW, Db, DX)



class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1,
                 padding: int = 0, bias:bool = True, store_grad:bool = False):

        super(Conv2D, self).__init__()

        self.weights = np.random.rand(out_channels, in_channels, *kernel_size)
        if bias:
            # bias initialised to zero
            self.bias = np.zeros((out_channels,1))

        self.ks, self.stride, self.pad = kernel_size, stride, padding

    def forward(self, input:np.ndarray) -> tuple:
        # forward pass for conv layer and stores cached variables for backward pass
        return  utils.correlation2d(input, self.weights, self.bias,
                                    stride=self.stride, padding=self.pad)

    def bwd(self, Dout:np.ndarray, cache:tuple) -> tuple:
        '''
        function to perform backward pass on conv layer
        :param Dout: np.ndarray, differential on the output of the conv layer. shape: NkxCxHxW
        :param cache: tuple, containes cached variables during forward pass for easy differentiation
        :return: (dW, db, dX) tuple of partial derivatives wrt to parameters W (kernel weights), b (bias) and input X
        '''

        input, transformed_input, kernel, bias, stride, padding = cache

        # get shapes
        B, C, H, W = input.shape  # B is the batch size (number of inputs)
        Nk, Ck, Hk, Wk = kernel.shape  # Nk is the number of kernels used.

        # gradients that have to be calculated in a con layer:
        #   - Db: Dout*dout/db partial derivative of out w.r.t. bias
        #   - DW: Dout*dout/dW partial derivative of our w.r.t. kernel weights
        #   - DX: Dout*dout/dX partial derivative of the output w.r.t. input

        # note that we are applying chain rule by multipling with Dout since this could be any inner layer,
        # hence the partial derivatives will depend on the differential of the following layers, which is assumed
        # to be contained in Dout. (this function will apply recursively for each layer, hence building up on the
        # gradients of the first layers in a backward fashion). Note that we simply did W*X + b, hence the partial
        # erivatives will simply be the ones of a linear combination.

        # partial of bias is simply del_ij, but recall that due to broadcasting we are adding bias element-wise to
        # each entry of W*X. this will lead to sum_k(Dout_ijkl*del_ij) -> the only dimension were there will not be a
        # sum is the number of channels of Dout, which is the same as the number of filters that were used (Nk).
        Db = np.sum(Dout, axis=(0,2,3))
        Db = Db.reshape(Nk,-1)

        # recall that we reshaped the input to perform fast convolution, we can also quickly get the gradients
        # as matrix multplication by taking the derivative of the matrix product instead of the convolution, hence we
        # need to reshape dout, which is assumed to have standard image shape NkxCxHxW to the same shape it has in
        # utils.correlation2D before being reshaped. We also the kernel for the same reasons.
        Dout = Dout.transpose(1,2,3,0).reshape(Nk,-1)
        kernel = kernel.reshape(Nk,-1)

        # partial of weights will be the input itself, which we multiply by Dout by chain rule
        DW = Dout @ transformed_input.T
        DW = DW.reshape(Nk, Ck, Hk, Wk) # we need to reshape it back to the original shape of the kernel

        # partial of inputs will simply be the kernel weights, which we again multiply by Dout by chain rule
        tranformed_DX = kernel.T @ Dout

        # we now need to do a reshaping procedure opposite to the one we did in utils.correlation2D, to get the image back
        # to its original shape
        DX = utils.neighborhoodMatrix2image(tranformed_DX, input.shape, kernel.shape, stride, padding)

        return (DW, Db, DX)



