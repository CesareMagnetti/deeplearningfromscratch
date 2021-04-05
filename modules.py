import utils
import numpy as np
from collections import OrderedDict

class Module():
    """
    Base class that all instanciated model/layer must inherit, it will set up the model/layer and
    handle basic functionalities.
    """

    def __init__(self):
        self.training = False # boolean variable to represent id module in training mode or evaluation mode
        self._parameters = OrderedDict() # store module's parameters
        self._caches = OrderedDict() # store forward pass caches to compute derivatives in backward pass
        self._backward_hooks = OrderedDict() # store the grad of the parameters when backward is called
        self._modules = OrderedDict() # store all sub-modules that will be contained in the model

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function not yet implemented for testing!')


    def bwd(self, *args, **kwargs):
        pass

    def backward(self, Dout = None):
        return self.bwd(Dout)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_parameter(self, name:str, param:np.ndarray) -> None:
        if name == '':
            raise ValueError('parameter name cannot be empty')
        else:
            assert isinstance(param, np.ndarray), "can only add instances of ``ndarray`` to self._parameters. Got {}".format(type(param))
            self._parameters[name] = param

    def add_cache(self, name:str, param:np.ndarray) -> None:
        if name == '':
            raise ValueError('cache name cannot be empty')
        else:
            assert isinstance(param, np.ndarray), "can only add instances of ``ndarray`` to self._caches. Got {}".format(type(param))
            self._caches[name] = param

    def add_module(self, name:str, module) -> None:
        if name == '':
            raise ValueError('module name cannot be empty')
        else:
            assert isinstance(module, Module), "can only add modules that inherit the ``Module`` class. Got {}".format(type(module))
            self._modules[name] = module


    def add_backward_hook(self, name:str, param:np.ndarray) -> None:
        if name == '':
            raise ValueError('backward hook name cannot be empty')
        else:
            assert isinstance(param, np.ndarray), "can only add instances of ``ndarray`` to self._backward_hooks. Got {}".format(type(param))
            self._backward_hooks[name] = param

    def update_params(self, Dout=None, lr = 0.001):
        """
        updates parameters using gradient descent
        :param loss: Module, loss class after a forward pass
        :param Dout: np.ndarray, current downstream gradient
        :param lr: learning rate, default 0.001
        """
        if self.training:
            if self._modules:
                for name in reversed(self._modules):
                    module = self._modules[name]
                    Dout = module.backward(Dout)
                    for param, hook in module._backward_hooks.items():
                        module._parameters[param] -= lr*hook
            else:
                _ = self.backward(Dout)
                for param, hook in self._backward_hooks.items():
                    self._parameters[param] -= lr * hook

    def zero_grad(self):
        self._caches = OrderedDict()
        self._backward_hooks = OrderedDict()


    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def named_parameters(self):
        if self._modules:
            return self.submodules()
        else:
            return self._parameters.items()

    def submodules(self):
        return self._modules.items()

    def reversed_submodules(self):
        reversed_modules = OrderedDict()
        for name in  reversed(self._modules):
            reversed_modules[name] = self._modules[name]
        return reversed_modules.items()

    def extra_repr(self):
        _name = "\n"
        if self._modules:
            for _, module in self.submodules():
                for key, item in module.named_parameters():
                    _name += "\t{}\t{}\n".format(key, item.shape)
                _name = "\n"
        else:
            for key, item in self.named_parameters():
                _name += "\t{}\t{}\n".format(key, item.shape)

        return _name


class Linear(Module):
    '''
    implementation of a linear layer and its backward
    '''

    def __repr__(self):
        _name = "Linear layer"
        return _name+self.extra_repr()

    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        # weights initialised from a centralized random distribution with std 0.1
        self.add_parameter("weight", np.random.normal(0, 0.1, size=(in_features, out_features)))
        # bias is initialised to zero
        self.add_parameter("bias", np.zeros(out_features))

    def forward(self, input: np.ndarray) -> tuple:
        weight, bias = self._parameters['weight'], self._parameters['bias']
        out = input @ weight + bias
        self.add_cache("input", input)
        return out

    def bwd(self, Dout:np.ndarray) -> np.ndarray:
        '''
        function to perform backward pass on linear layer
        :param Dout: np.ndarray, differential on the output w.r.t. input layer.
        :return: dX: np.ndarray, gradient wrt to the input of the layer, will be used by previous layers as
                     Dout (input for backwards). DW and Db will be stored in the _backward_hooks attribute,
                     which will update the layers parameters when needed.
        '''

        # partials are trivial since linear layer simply performs XW+b, note that b is broadcasted element-wise and that
        # this can be a general inner layer, hence derivatives depend on Dout due to chain rule.
        input = self._caches['input']
        DW = input.T @ Dout #d[XW+b}/dW = X then inner product with Dout due to chain rule.
        Db = np.sum(Dout, axis=0) #d[XW+b]/db = 1 then multiply by Dout due to chain rule.
                                  # sum over first dimension because b was broadcasted element-wise

        weight, bias = self._parameters['weight'], self._parameters['bias']
        DX = Dout @ weight.T #d[XW+b}/dX = W then inner product with Dout due to chain rule.

        # store gradients of the layers parameters
        self.add_backward_hook("weight", DW)
        self.add_backward_hook("bias", Db)

        return DX



class Conv2D(Module):
    """
    An implementation of a convolutional layer.

    The input consists of B data points, each with C channels, height H and
    width W. We convolve each input with Nk different filters, where each filter
    spans all C channels and has height Hk and width Wk.

    Parameters:
    - w: Filter weights of shape (Nk, C, Hk, Wk)
    - b: Biases, of shape (Nk,)
    - kernel_size: Size of the convolving kernel
    - stride: The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    - padding: The number of pixels that will be used to zero-pad the input.
    """

    def __repr__(self):
        _name = "Conv2D layer"
        return _name+self.extra_repr()

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0):

        super(Conv2D, self).__init__()

        # weights initialised from a centralised random distribution with std 0.1
        self.add_parameter("weight", np.random.normal(0, 0.1, size=(out_channels, in_channels, *kernel_size)))
        # bias is initialised to zero
        self.add_parameter("bias", np.zeros((out_channels,1)))

        self.ks, self.stride, self.pad = kernel_size, stride, padding

    def forward(self, input:np.ndarray) -> tuple:
        '''
        see comments of utils.correlation2d() for insights on the forward pass of conv2D
        :param input: np.ndarray, input to the conv2D layer
        :return: out: np.ndarray, output to the conv2D layer
        '''
        weight, bias = self._parameters['weight'], self._parameters['bias']
        # forward pass for conv layer and stores cached variables for backward pass
        out, caches = utils.correlation2d(input, weight, bias, stride=self.stride, padding=self.pad)
        for key, cache in caches.items():
            self.add_cache(key, cache)
        return out

    def bwd(self, Dout:np.ndarray) -> np.ndarray:
        '''
        function to perform backward pass on conv layer
        :param Dout: np.ndarray, differential on the output of the conv layer. shape: NkxCxHxW
        :return: dX: np.ndarray, gradient wrt to the input of the layer, will be used by previous layers as
                     Dout (input for backwards). DW and Db will be stored in the layer parameters .grad attribute,
                     which will update the layers parameters when needed.
        '''

        input, transformed_input = self._caches['input'], self._caches['transformed_input']
        weight, bias = self._parameters['weight'], self._parameters['bias']

        Nk, Ck, Hk, Wk = weight.shape  # Nk is the number of kernels used.

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
        kernel = weight.reshape(Nk,-1)

        # partial of weights will be the input itself, which we multiply by Dout by chain rule
        DW = Dout @ transformed_input.T
        DW = DW.reshape(Nk, Ck, Hk, Wk) # we need to reshape it back to the original shape of the kernel

        # partial of inputs will simply be the kernel weights, which we again multiply by Dout by chain rule
        tranformed_DX = kernel.T @ Dout

        # we now need to do a reshaping procedure opposite to the one we did in utils.correlation2D, to get the image back
        # to its original shape
        DX = utils.neighborhoodMatrix2image(tranformed_DX, input.shape, weight.shape, self.stride, self.pad)

        # store gradients of the layers parameters
        self.add_backward_hook("weight", DW)
        self.add_backward_hook("bias", Db)

        return DX

class View(Module):
    """
    class to reshape an input to go through linear layers and include it in backprop
    """
    def __repr__(self):
        return "View tensor\n"

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        reshaped_input = input.reshape(self.shape)
        self.add_cache("input", input)
        self.add_cache("reshaped_input", reshaped_input)
        return reshaped_input

    def bwd(self, Dout):
        input, reshaped_input = self._caches["input"], self._caches["reshaped_input"]
        Dout = Dout.reshape(input.shape)
        return Dout

class ReLU(Module):
    """
    rectified linear unit activation function: out = max(0,X)
    """

    def __repr__(self):
        _name = "ReLU activation\n"
        return _name

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        mask = input <= 0
        self.add_cache("mask", mask)
        input[mask] = 0
        return input

    def bwd(self, Dout):
        mask = self._caches["mask"]
        DX = Dout.copy()
        DX[mask] = 0
        return DX


class Sigmoid(Module):
    """
    sigmoid activation function: out = 1/[1+exp(-X)]
    """

    def __repr__(self):
        _name = "Sigmoid activation\n"
        return _name

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        '''
        aplying sigmoid forward pass (storing necessary caches)
        :param input: np.ndarray, input to the softmax layer
        :return: out: np.ndarray, output to the softmax layer
        '''
        out = 1/(1+np.exp(-input))
        self.add_cache("out", out)
        return out

    def bwd(self, Dout):
        out = self._caches["out"]
        return Dout*out*(1-out)

class Softmax(Module):
    """
    sigmoid activation function: out_j = exp(x_j)/[sum_i (exp(-x_i)]
    """

    def __repr__(self):
        _name = "Softmax activation\n"
        return _name

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        '''
        aplying softmax forward pass (storing necessary caches)
        :param input: np.ndarray, input to the softmax layer
        :return: out: np.ndarray, output to the softmax layer
        '''
        e = np.exp(input - np.max(input))
        s = np.sum(e, axis=1, keepdims=True)
        softmax = e / s
        # cache variables for backward pass
        self.add_cache("input", input)
        self.add_cache("softmax", softmax)
        return softmax

    def bwd(self, Dout):
        # the derivative of a softmax can be derived to be dS(x)/dx = Sj*(del_jk - Sk), where S(x) is the softmax of x
        # and del_jk is the kronecker delta function.

        input, softmax = self._caches["input"], self._caches["softmax"]
        B,N = input.shape

        # expand above equation to Sj*del_ij - Sj*Sk, where both Sj and Sk are 2D matrices BxN where N is the number of neurons.
        # Sj*Sk is an outer product of S with itself
        Sj_Sk = np.einsum('ij,ik->ijk', softmax, softmax)  # BxNxN

        #Sj*del_jk
        Sj_deljk = np.einsum('ij,jk->ijk', softmax, np.eye(N, N))  # BxNxN

        dSoftmax = Sj_deljk - Sj_Sk

        # we multiply the derivative of the softmax w.r.t input by downstream gradient Dout (chain rule)
        Dout = np.einsum('ijk,ik->ij', dSoftmax, Dout)  # BxN
        return Dout

class NegativeLogLikelyhood(Module):
    """
    Negative Log Likelyhood Loss for single class MLE fit, assuming labels as integers starting from 0
    """

    def __repr__(self):
        return "Negative Log-Likelyhood loss\n"

    def __init__(self):
        super(NegativeLogLikelyhood, self).__init__()

    def forward(self, input, label):
        B = input.shape[0]
        # approximate full loss with batch loss (SGD)
        NLL = -np.mean(np.log(input[np.arange(B), label]))
        self.add_cache("input", input)
        self.add_cache("label", label)
        return NLL

    def bwd(self, Dout=None):
        input, label = self._caches["input"], self._caches["label"]
        B = input.shape[0]
        Dout = np.zeros(input.shape)
        Dout[np.arange(B), label] = -1/input[np.arange(B), label]
        return Dout

