from utils import *

class Param():
    def __init__(self, data:np.ndarray=None, store_grad:bool = False):
        self.data = data
        if store_grad:
            self.grad = np.zeros_like(data)

    def set_data(self, data:np.ndarray, store_grad:bool = False):
        self.data = data
        if store_grad:
            self.grad = np.zeros_like(data)


class Module():
    def __init__(self):
        self.weights = Param()
        self.bias = Param()

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function not yet implemented!')

    def bwd(self, *args, **kwargs):
        raise NotImplementedError('bwd (backward implementation) method not yet implemented!'
                                  )
    def backward(self):
        return self.bwd(self.out)

    def __call__(self, *args, **kwargs):
        self.out = self.forward(*args, **kwargs)
        return self.out



class Linear(Module):
    '''
    implementation of a linear layer which simply implements W*input + bias and evaluates gradients once called pass
    '''
    def __init__(self, in_features: int, out_features: int, store_grad:bool = False):
        super(Linear, self).__init__()
        self.weights.set_data(np.random.rand(in_features, out_features), store_grad)
        # bias is initialised to zero
        self.bias.set_data(np.zeros(out_features), store_grad)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs.shape)>1:
            out = []
            for _input in inputs:
                out.append(self.weights.data.dot(_input) + self.bias.data)
            return np.array(out)
        else:
            return self.weights.data.dot(inputs) + self.bias.data



class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1,
                 padding: int = 0, bias:bool = True, store_grad:bool = False):

        super(Conv2D, self).__init__()

        self.weights.set_data(np.random.rand(out_channels, in_channels, *kernel_size), store_grad)
        if bias:
            # bias initialised to zero
            self.bias.set_data(np.zeros(out_channels), store_grad)

        self.ks, self.stride, self.pad = kernel_size, stride, padding

    def forward(self, input:np.ndarray) -> np.ndarray:

        # convolve input
        if self.bias.data is not None:
            return correlation2d(input, self.weights.data, stride=self.stride, padding=self.pad) + self.bias.data
        else:
            return correlation2d(input, self.weights.data,stride=self.stride, padding=self.pad)
