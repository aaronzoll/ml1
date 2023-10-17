import numpy as np 
import pandas as pd 
import time as t
import math as math
import matplotlib.pyplot as plt 


Num=100000
X=np.linspace(2,Num+2,101)
print(X)
Y=[]
for i in X:
    N=int(i)
    L=range(1,N+1)
    M=np.mod(N,L)
    val=M.sum()/N**2
    Y.append(val)
plt.plot(X,Y,'o')
plt.show()
