import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fc import FCLayer
from utils import *


class BP(object):
    """
    A implement of BP neural network
    """
    def __init__(self):
        self.fc1 = FCLayer(1, 10)
        self.fc2 = FCLayer(10, 1)

    def forward(self, x):
        x = self.fc1.forward(x)
        self.b = tanh(x) #active function
        x = self.fc2.forward(self.b)
        self.y_ = x.squeeze()
        return self.y_

    def backward(self, label, lr=0.0001, weight_decay=0.04):
        grad = 2*(self.y_ - label)
        grad = np.array([grad])
        grad = grad[:,np.newaxis] #1x1
        self.fc2.gradient(grad) 
        grad = np.dot(self.fc2.weights, grad) #10x1 x 1x1 = 10x1
        self.b = self.b.T
        grad *= (1-self.b**2) # d(tanh x)/dx = 1 - (tanh x)^2
        self.fc1.gradient(grad)

        self.fc2.backward(lr, weight_decay)
        self.fc1.backward(lr, weight_decay)


def train(net):
    df1 = pd.read_table(r"D:\Coding\Pycharm\mlp\regression_train.txt", header=None, names=['x','y'], sep=' ')
    X = np.array(df1['x'])
    Y = np.array(df1['y'])

    length = len(Y)
    train_X = X[0:450]
    train_Y = Y[0:450]
    val_X = X[450:-1]
    val_Y = Y[450:-1]

    
    LR = 0.0001
    WEIGHT_DECAY = 0 
    EPOCH = 300

    train_losses = []
    val_losses = []
    for epoch in range(EPOCH):
        loss = 0
        for x, y in zip(train_X, train_Y):
            x = np.array([x])
            x = x[:,np.newaxis].T
            output = net.forward(x)
            loss += mse_loss(output, y)
            net.backward(y, LR, WEIGHT_DECAY)
        print("Epoch %d train loss: %.4f" % (epoch, loss/len(train_Y)))
        train_losses.append(loss/len(train_Y))

        loss = 0
        for x, y in zip(val_X, val_Y):
            x = np.array([x])
            x = x[:,np.newaxis].T
            output = net.forward(x)
            loss += mse_loss(output, y)
        val_losses.append(loss/len(val_Y))
        print("Epoch %d validation loss: %.4f" % (epoch, loss/len(val_Y)))


    plt.figure()
    plt.title("Loss")
    l1, = plt.plot(np.linspace(0,EPOCH,EPOCH), train_losses, 'r')
    l2, = plt.plot(np.linspace(0,EPOCH,EPOCH), val_losses, 'b')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.show()
    

def pred(net):
    df1 = pd.read_table(r"D:\Coding\Pycharm\mlp\regression_pred.txt", header=None, names=['x'], sep=' ')
    X = np.array(df1['x'])
 
    prediction = []
    for x in X:
        x = np.array([x])
        x = x[:,np.newaxis].T
        output = net.forward(x)
        prediction.append(output)
    
    with open(r"D:\Coding\Pycharm\mlp\regression_pred_result.txt", "w+") as f:
        for i in range(len(prediction)):
            prediction[i] = str(prediction[i]) + "\n"
        f.writelines(prediction)

if __name__=='__main__':
    net = BP()
    train(net)
    print("Now start to predict")
    pred(net)
    print("Done")

