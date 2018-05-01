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
        self.fc1 = FCLayer(2, 10)
        self.fc2 = FCLayer(10, 3)

    def forward(self, x):
        x = self.fc1.forward(x)
        self.b = tanh(x) #active function
        x = self.fc2.forward(self.b)
        x = x.squeeze()
        self.y_ = softmax(x) #active function
        return self.y_

    def backward(self, label, lr=0.0001, weight_decay=0.04):
        grad = self.y_ - label
        grad = grad[:,np.newaxis] #3x1
        self.fc2.gradient(grad) 
        grad = np.dot(self.fc2.weights, grad) #10x3 x 3x1 = 10x1
        self.b = self.b.T
        grad *= (1-self.b**2) # d(tanh x)/dx = 1 - (tanh x)^2
        self.fc1.gradient(grad)

        self.fc2.backward(lr, weight_decay)
        self.fc1.backward(lr, weight_decay)

def one_hot(y, k):
    vec = np.zeros(k)
    vec[int(y)] = 1
    return vec

def train(net):
    df1 = pd.read_table(r"D:\Coding\Pycharm\mlp\classfication_train.txt", header=None, names=['x1','x2','y'], sep=' ')
    x1 = np.array(df1['x1'])
    x2 = np.array(df1['x2'])
    Y = np.array(df1['y'])
    X = np.column_stack((x1,x2))

    length = len(Y)
    train_X = X[0:500]
    train_Y = Y[0:500]
    val_X = X[500:-1]
    val_Y = Y[500:-1]

    
    LR = 0.01
    WEIGHT_DECAY = 0 
    EPOCH = 1000

    train_accu = []
    val_accu = []
    losses = []
    for epoch in range(EPOCH):
        loss = 0
        train_correct = 0
        for x, y in zip(train_X, train_Y):
            x = x[:,np.newaxis].T
            output = net.forward(x)
            pred = np.argmax(output)
            train_correct += (int(pred) == int(y))
            y = one_hot(y, 3)
            loss += cross_entropy_loss(output, y)
            net.backward(y, LR, WEIGHT_DECAY)
        print("Epoch %d loss: %.4f" % (epoch, loss))
        print("Epoch %d Train Accu: %.2f" % (epoch, train_correct/len(train_Y)))
        train_accu.append(train_correct/len(train_Y))
        losses.append(loss)

        val_correct = 0
        for x, y in zip(val_X, val_Y):
            x = x[:,np.newaxis].T
            output = net.forward(x)
            pred = np.argmax(output)
            val_correct += (int(pred) == int(y))
        print("Epoch %d Validation Accu: %.2f" % (epoch, val_correct/len(val_Y)))
        val_accu.append(val_correct/len(val_Y))

    plt.figure()
    plt.title("loss")
    plt.plot(np.linspace(0,EPOCH,EPOCH), losses)
    plt.xlabel('epoch')
    plt.ylabel('cross entropy loss')
    plt.show()

    plt.figure()
    plt.title("Accu")
    l1, = plt.plot(np.linspace(0,EPOCH,EPOCH), train_accu, 'r')
    l2, = plt.plot(np.linspace(0,EPOCH,EPOCH), val_accu, 'b')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.show()
    

def pred(net):
    df1 = pd.read_table(r"D:\Coding\Pycharm\mlp\classfication_pred.txt", header=None, names=['x1','x2'], sep=' ')
    x1 = np.array(df1['x1'])
    x2 = np.array(df1['x2'])
    X = np.column_stack((x1,x2))
    prediction = []
    for x in X:
        x = x[:,np.newaxis].T
        output = net.forward(x)
        pred = np.argmax(output)
        prediction.append(pred)
    
    with open(r"D:\Coding\Pycharm\mlp\classfication_pred_result.txt", "w+") as f:
        for i in range(len(prediction)):
            prediction[i] = str(prediction[i]) + "\n"
        f.writelines(prediction)

if __name__=='__main__':
    net = BP()
    train(net)
    print("Now start to predict")
    pred(net)
    print("Done")

