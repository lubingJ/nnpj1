from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / (in_dim + out_dim))  # Xavier初始化
        self.W = np.random.normal(0, scale, size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}
        self.isReLU = False 

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        self.W = self.params['W']
        self.b = self.params['b']
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        input_grad = np.dot(grad, self.W.T) 
        self.grads['W'] = np.dot(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0)
        return input_grad

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))  # He初始化
        self.W = initialize_method(0, scale, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)

        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_channels, in_height, in_width = X.shape

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        cols = im2col(X_padded, self.kernel_size, self.stride)
        W_col = self.W.reshape(self.out_channels, -1)

        output = np.dot(cols, W_col.T) + self.b
        return output.reshape(batch_size, out_height, out_width, self.out_channels).transpose(0, 3, 1, 2)

    
    def backward(self, grad):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        batch_size, in_channels, in_height, in_width = X.shape
        _, out_channels, out_height, out_width = grad.shape

        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        input_grad = np.zeros_like(X)

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                              (self.padding, self.padding)), mode='constant')
            input_grad_padded = np.zeros((batch_size, in_channels, 
                                    in_height + 2*self.padding, 
                                    in_width + 2*self.padding))
        else:
            X_padded = X
            input_grad_padded = np.zeros_like(X)
    
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)    

        cols = im2col(X_padded, self.kernel_size, self.stride)
        self.grads['W'] = np.dot(grad_reshaped.T, cols).reshape(self.W.shape)
        self.grads['b'] = np.sum(grad_reshaped, axis=0)

        W_reshaped = self.W.reshape(out_channels, -1)
        input_cols = np.dot(grad_reshaped, W_reshaped)
    
        input_grad = col2im(input_cols, X_padded.shape, self.kernel_size, self.stride)
    
        if self.padding > 0:
            input_grad = input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
    
        return input_grad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False
        self.isReLU = True

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        batch_size = predicts.shape[0]
        self.labels = labels
        self.probs = predicts
        max_vals = np.max(predicts, axis=1, keepdims=True)
        exp_vals = np.exp(predicts - max_vals)  
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        probs_clipped = np.clip(probs, 1e-15, 1.0)
        loss = -np.sum(np.log(probs_clipped[np.arange(batch_size), labels])) / batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        batch_size = self.probs.shape[0]

        max_vals = np.max(self.probs, axis=1, keepdims=True)
        exp_vals = np.exp(self.probs - max_vals)  
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        grads = probs.copy()
        grads[np.arange(batch_size), self.labels] -= 1
        grads /= batch_size
        self.grads = grads
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def __call__(self, X):
        return self.foward(X)
    
    def foward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad):
        return grad.reshape(self.input_shape)

class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

def im2col(X, kernel_size, stride):
    batch_size, channels, height, width = X.shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    cols = np.zeros((batch_size, channels, kernel_size, kernel_size, out_h, out_w))

    for h in range(kernel_size):
        h_end = h + out_h * stride
        for w in range(kernel_size):
            w_end = w + out_w * stride
        cols[:, :, h, w, :, :] = X[:, :, h:h_end:stride, w:w_end:stride]
    
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)

def col2im(cols, img_shape, kernel_size, stride):
    batch_size, channels, height, width = img_shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    
    img = np.zeros(img_shape)
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            img[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size] += \
                cols[i*out_w + j :: out_h*out_w, :].reshape(batch_size, channels, kernel_size, kernel_size)
    return img
