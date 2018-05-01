import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loss_func(x,y,theta):
    return sum((theta.dot(x.T) - y)**2)/2

def gd(x, y, theta, dim, size, times, alpha):
    loss = np.zeros(times)
    for i in range(times):
        sum = np.zeros(dim+1)
        for j in range(size):
            temp = theta.dot(x[j]) - y[j]
            for k in range(dim):
                sum[k] += temp*x[j][k]
        
        for k in range(dim):    
            theta[k] -= alpha*sum[k]
        
        loss[i] = loss_func(x,y,theta)

    return loss

if __name__=='__main__':
    df = pd.read_table("linear_regression2.txt", header=None, names=['x1','x2','x3','x4','x5','y'],sep=' ')
    theta = np.zeros(6)
    x1 = np.array(df['x1'])
    x2 = np.array(df['x2'])
    x3 = np.array(df['x3'])
    x4 = np.array(df['x4'])
    x5 = np.array(df['x5'])
    x = np.column_stack((x1,x2,x3,x4,x5))
    x = np.column_stack((x,np.ones(len(x))))
    y = np.array(df['y'])
    times = 200
    lr = 0.000000000001

    loss = gd(x, y, theta, 5, len(x), times, lr)
    print(theta)
    plt.figure()
    plt.title("loss")
    plt.plot(np.linspace(0,times,times),loss)
    plt.show()