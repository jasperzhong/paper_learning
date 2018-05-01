import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(x, theta):
    return 1.0/(1+np.exp(-theta.dot(x.T)))

def loss_func(x,y,theta):
    val = sigmoid(x,theta)
    return sum(-y * np.log(val) - (1-y) * np.log(1-val))

def gd(x, y, theta, dim, size, times, alpha):
    loss = np.zeros(times)
    for i in range(times):
        sum = np.zeros(dim+1)
        for j in range(size):
            temp = sigmoid(x[j],theta)
            for k in range(dim):
                sum[k] += (-y[j]*(1-temp) + (1-y[j])*temp)*x[j][k]
        
        for k in range(dim):    
            theta[k] -= alpha*sum[k]
        
        loss[i] = loss_func(x,y,theta)

    return loss

if __name__=='__main__':
    df = pd.read_table("logistic_regression2.txt", header=None, names=['x1','x2','x3','x4','y'],sep=' ')
    theta = np.zeros(5)
    x1 = np.array(df['x1'])
    x2 = np.array(df['x2'])
    x3 = np.array(df['x3'])
    x4 = np.array(df['x4'])
    
    x = np.column_stack((x1,x2,x3,x4))
    x = np.column_stack((x,np.ones(len(x))))
    y = np.array(df['y'])
    times = 100
    lr = 0.00005
    loss = gd(x, y, theta, 4, len(x), times, lr)
    print(theta)
    
    
    plt.figure()
    plt.title("loss")
    plt.plot(np.linspace(0,times,times),loss)
    plt.show()