
import numpy as np
from scipy import linalg

# you pass the X and the Y of the function and the degree of the polinomy
# you want and you get the coefficent in ascent order of degree

def polinomialRegression(X,Y,degree=2):
    degree += 1
    #creating B
    B = np.zeros((degree,degree))
    B[0,0] = len(Y)

    for i1,x in enumerate(B):
        for i2,y in enumerate(x[:degree-i1]):
            B[i1,i1+i2] = np.sum(X**(2*i1+i2))

    B += np.transpose(B)
    B[np.eye(degree) == 1] /= 2

    #creating C
    C = np.zeros((degree,1))
    for i,x in enumerate(C):
        C[i] = np.sum(np.multiply(Y,X**i))

    #solving AB=C for A
    A = linalg.solve(B,C)

    return A