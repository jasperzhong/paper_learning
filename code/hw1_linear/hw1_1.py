import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loss_func(x,y,theta):
    return sum((theta.dot(x.T) - y)**2)/2

def gd(x, y, theta, dim, size, times, alpha):
    loss = np.zeros(times)
    for i in range(times):
        sum = np.zeros(dim)
        for j in range(size):
            temp = theta.dot(x[j]) - y[j]
            for k in range(dim):
                sum[k] += temp*x[j][k]
        
        for k in range(dim):    
            theta[k] -= alpha*sum[k]
        
        loss[i] = loss_func(x,y,theta)

    return loss

if __name__=='__main__':
    df = pd.read_table("linear_regression1.txt", header=None, names=['x','y'],sep=' ')
    theta = np.array([0.0 ,0.0])
    x = np.array(df['x'])
    x = x / np.max(x)
    x = np.column_stack((x,np.ones(len(x))))
    y = np.array(df['y'])
    times = 100
    lr = 0.0003
    loss = gd(x, y, theta, 2, len(x), times, lr)
    print(theta)
    pred = theta.dot(x.T)
    plt.figure()
    plt.title("regression")
    plt.scatter(x[:,0],y)
    plt.plot(x[:,0],pred,'r',lw=3)
    plt.show()

    plt.figure()
    plt.title("loss")
    plt.plot(np.linspace(0,times,times),loss)
    plt.show()