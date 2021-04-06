# DeepLearningFromScratch

personal project to try and really get an understanding on how common libraries such as Pytorch/TensorFlow etc work. Not going in detail, nor fixating on efficiency (using numpy whenever possible), this was a great project to gain a better understanding on what's really happening behind pytorch interface. Tested code training a simple CNN on MNIST, achieving approximately 96.5%% accuracy on test set (way better than expected), see the python notebook for the training experiment. 

May continue this in the future with implementations of further models.

## Usage

1. clone repo and create virtual environment (Python 3.8.2)
```bash
git clone git@gitlab.com:cesare.magnetti/deeplearningfromscratch.git <your_working_directory>
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. You can create a custom CNN using a pytorch-like framework (not the best code interface but it seems to work). example model:
``` python
class Net(Module):
    def __repr__(self):
        _name = "Custom Network\n"
        for _, module in self.submodules():
            _name+=module.__repr__()
        return _name
    
    def __init__(self):
        super(Net, self).__init__()
        self.add_module("conv1", modules.Conv2D(in_channels=1,
                                                out_channels=4,
                                                kernel_size=(3,3),
                                                padding=1,
                                                stride=2))
        
        self.add_module("relu1", modules.ReLU())
        
        self.add_module("conv2", modules.Conv2D(in_channels=4,
                                                out_channels=8,
                                                kernel_size=(3,3),
                                                padding=1,
                                                stride=2))
        self.add_module("relu2", modules.ReLU())
        
        self.add_module("conv3", modules.Conv2D(in_channels=8,
                                        out_channels=16,
                                        kernel_size=(4,4),
                                        padding=0,
                                        stride=1))
        self.add_module("relu3", modules.ReLU())
        
        self.add_module("view", modules.View(shape=(-1, 16*4*4)))
        
        self.add_module("linear", modules.Linear(16*4*4, 10))
        
        self.add_module("softmax", modules.Softmax())
        
        self.add_module("NLL", modules.NegativeLogLikelyhood())
        
    
    def forward(self, inputs, labels):
        out = self._modules['relu1'](self._modules['conv1'](inputs))
        out = self._modules['relu2'](self._modules['conv2'](out))
        out = self._modules['relu3'](self._modules['conv3'](out))
        out = self._modules['view'](out)
        out = self._modules['linear'](out)
        out = self._modules['softmax'](out)
        loss = self._modules['NLL'](out, labels)
        return out, loss 
```

3. You can train the model in a simple loop fashion (CPU only) (uses gradient descent at the minibatch level (minibatch SGD))

```python
model = Net()
print(model)

# some hyper params
max_epochs = 2
batch_size = 32
learning_rate = 0.001

# train
model.train() #activates gradients
for epoch in range(1,max_epochs+1):
    for idx, (img, lbl) in enumerate(batch_data(train_data, batch_size)):
        model.zero_grad()
        out, loss = model(img, lbl)
        model.update_params(lr = learning_rate)
```

## Contributors
@cesare.magnetti



