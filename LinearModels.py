import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


def draw_line(coef,intercept, mi, ma):
    # for the separating hyper plane ax+by+c=0, the weights are [a, b] and the intercept is c
    # to draw the hyper plane we are creating two points
    # 1. ((b*min-c)/a, min) i.e ax+by+c=0 ==> ax = (-by-c) ==> x = (-by-c)/a here in place of y we are keeping the minimum value of y
    # 2. ((b*max-c)/a, max) i.e ax+by+c=0 ==> ax = (-by-c) ==> x = (-by-c)/a here in place of y we are keeping the maximum value of y
    points=np.array([[((-coef[1]*mi - intercept)/coef[0]), mi],[((-coef[1]*ma - intercept)/coef[0]), ma]])
    plt.plot(points[:,0], points[:,1])


# here we are creating 2d imbalanced data points 
ratios = [(100,2), (100, 20), (100, 40), (100, 80)]
alphas = [0.001,1,100]
plt.figure(figsize=(20,5))

#https://stackoverflow.com/questions/46511017/plot-hyperplane-linear-svm-python

def Plane(X, Y, alpha):
    svm = SVC(C=alpha, kernel='linear').fit(X,Y)
    # w.x + b = 0
    w = svm.coef_[0]
    b = -w[0] / w[1]
    y_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    draw_line(w,b, y_min, y_max)

for j,i in enumerate(ratios):
    plt.subplot(1, 4, j+1)
    X_p=np.random.normal(0,0.05,size=(i[0],2))
    X_n=np.random.normal(0.13,0.02,size=(i[1],2))
    y_p=np.array([1]*i[0]).reshape(-1,1)
    y_n=np.array([0]*i[1]).reshape(-1,1)
    X=np.vstack((X_p,X_n))
    y=np.vstack((y_p,y_n))
    
    for alpha in alphas:
        Plane(X,y,alpha)
        plt.scatter(X_p[:,0],X_p[:,1])
        plt.scatter(X_n[:,0],X_n[:,1],color='red')
        plt.show()


#you can start writing code here.
from sklearn.linear_model import LogisticRegression

def LRPlane(X, Y, alpha):
    svm = LogisticRegression(C=alpha).fit(X,Y)
    # w.x + b = 0
    w = svm.coef_[0]
    b = -w[0] / w[1]
    y_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    draw_line(w,b, y_min, y_max)

for j,i in enumerate(ratios):
    plt.subplot(1, 4, j+1)
    X_p=np.random.normal(0,0.05,size=(i[0],2))
    X_n=np.random.normal(0.13,0.02,size=(i[1],2))
    y_p=np.array([1]*i[0]).reshape(-1,1)
    y_n=np.array([0]*i[1]).reshape(-1,1)
    X=np.vstack((X_p,X_n))
    y=np.vstack((y_p,y_n))
    
    for alpha in alphas:
        LRPlane(X,y,alpha)
        plt.scatter(X_p[:,0],X_p[:,1])
        plt.scatter(X_n[:,0],X_n[:,1],color='red')
        plt.show()