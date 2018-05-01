import numpy as np

class FCLayer(object):
    """
    Fully connected layer (FC)
    """
    def __init__(self, in_features, out_features):
        """init """
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.standard_normal(size=(in_features, out_features))
        self.bias = np.random.standard_normal(size=(1, out_features))

        self.w_grad = np.zeros(self.weights.shape) #weights gradient
        self.b_grad = np.zeros(self.bias.shape) #bias gradient

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def gradient(self, grad):
        self.w_grad = np.dot(grad, self.x).T
        self.b_grad = grad.T

    def backward(self, lr=0.0001, weight_decay=0.04):
        """ backward propagation """
        self.weights *= 1 - weight_decay
        self.bias *= 1 - weight_decay
        self.weights -= lr * self.w_grad
        
        self.bias -= lr * self.b_grad

        self.w_grad = np.zeros(self.w_grad.shape)  # clear
        self.b_grad = np.zeros(self.b_grad.shape)  # clear


