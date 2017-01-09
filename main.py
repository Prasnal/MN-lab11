from math import *
import random
from scipy.fftpack import *
import numpy as np
import matplotlib.pyplot as plt
import time

def y0(i,n):
    omega=4*pi/n
    return cos(omega*i)+cos(2*omega*i)+cos(3*omega*i)

def random_var():
    X=random.random()
    Y=random.random()
    sign=1 if Y>0.5 else -1
    a=2*sign*X
    return a

def res(k):
    n=pow(2,k)
    i,xp=0,[]
    while i<n:
        xp.append(y0(i,n))
        i+=1
    return xp

def noise(k):
    n=pow(2,k)
    i,y=0,[]
    while i<n:
        y.append(random_var()+y0(i,n))
        i+=1
    return y

def discrimination(array):
    maximum=max(array)
    minimum=0.25*maximum
    result=[x if x>minimum else 0 for x in array]
    return result


#---------MAIN-----------------
k=10

#COSINE TRANSFORMATION
x=np.array(noise(k))

start=time.time()
result=dct(x,norm='ortho')
end=time.time()
print("time for cosine transformation:",end-start)

disc=discrimination(result)

start=time.time()
np.fft.fft(x)
end=time.time()
print("time for furier transformation:",end-start)

#REVERSE COSINE TRANSFORMATION
result2=idct(disc,norm='ortho')

#PLOT a
x2=np.linspace(0,pow(2,k),1025)

plt.plot(noise(k),'b') 
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('a.png')
plt.clf()

#PLOT b
plt.plot(result,'r')
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('b.png')
plt.clf()

#PLOT c
plt.plot(result[:20],'y')
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('c.png')
plt.clf()

#PLOT d
plt.plot(res(k),'r')
plt.plot(result2,'b')
plt.savefig('d.png')
plt.clf()
