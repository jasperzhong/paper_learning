from mpl_toolkits.mplot3d import Axes3D  
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
    df = pd.read_table("logistic_regression1.txt", header=None, names=['x1','x2','y'],sep=' ')
    theta = np.array([0.0 ,0.0, 0.0])
    x1 = np.array(df['x1'])
    x2 = np.array(df['x2'])
    x = np.column_stack((x1,x2))
    x = np.column_stack((x,np.ones(len(x))))
    y = np.array(df['y'])
    times = 100
    lr = 0.0003
    loss = gd(x, y, theta, 2, len(x), times, lr)
    print(theta)
    

    plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("regression")
    
    #plt.plot(x[:,0],pred,'r',lw=3)
    X1 = np.linspace(-2,6,1000)
    X2 = np.linspace(-2,6,1000)
    X1, X2 = np.meshgrid(X1,X2)
    Z = np.zeros((1000,1000))
    for i in range(1000):
        Z[:,i] = sigmoid(np.column_stack((X1[:,i],X2[:,i],np.ones(1000))),theta)
    ax.plot_surface(X1,X2,Z,cmap='rainbow')
    ax.scatter3D(x[:,0],x[:,1],y)
    plt.show()

    
    plt.figure()
    plt.title("loss")
    plt.plot(np.linspace(0,times,times),loss)
    plt.show()